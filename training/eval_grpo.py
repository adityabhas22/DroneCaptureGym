"""Evaluate a base / SFT / GRPO variant on the held-out DroneCaptureOps tasks.

This is the lightweight, vLLM-free counterpart to ``training/eval_policy.py``.
It uses the same model / adapter loading recipe as the GRPO trainer
(``transformers.AutoModelForCausalLM`` + ``peft.PeftModel``) and drives the
existing ``RolloutRunner`` through ``HFGeneratePolicy`` so the chat template,
trimming, and parsing all match what the model saw during training.

Why a new module instead of extending ``eval_policy.py``:

  - ``eval_policy.py`` hard-binds ``vllm.lora.LoRARequest`` and ``VLLMEngine``,
    both of which are exactly what we just removed from GRPO because of the
    repeated H200 / fabric-manager / OOM blockers documented in the GRPO
    branch overview.
  - We want one entrypoint that can score the *base model alone*, the
    *SFT-only adapter*, and a GRPO-trained adapter using the same generate
    backend. The single source of truth is the new HF-generate policy.
  - Eval needs to run inside an HF Job that already installs the GRPO deps
    (transformers + peft + trl + accelerate) — adding vLLM only for eval
    would re-introduce all the failure modes we just eliminated.

CLI:

    python -m training.eval_grpo \
        --base-model Qwen/Qwen3-4B-Instruct-2507 \
        --variant base \
        --variant sft:adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint \
        --variant grpo_kl005:rachit-suresh/dronecaptureops-grpo-fixspan-4b-h200-kl005:output/final_10/adapter \
        --seeds-per-task 2 \
        --output-dir artifacts/eval/

The adapter spec format is ``<repo_id>:<subfolder>`` (preferred for HF Hub
entries with checkpoints under a subfolder) or a plain repo id / local path.
``base`` is reserved and means "no adapter".
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LOG = logging.getLogger("dronecaptureops.eval_grpo")


# Held-out task list lives here so eval_policy.py and this module agree
# even when one of them is updated.
DEFAULT_HELD_OUT = [
    "scheduled_crane_window_wait_or_detour",
    "honest_partial_report_open_items",
    "strict_severity_weighted_triage",
    "edge_row_quality_bar",
    "privacy_safe_alternate_evidence",
    "return_margin_decision_point",
    "route_replan_when_primary_viewpoint_blocked",
]


# ---------------------------------------------------------------------------
# Variant parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantSpec:
    """One eval variant: a name and an optional adapter spec.

    ``adapter_spec`` is the raw string the user passed (``None`` for the
    base model). When non-None, it is forwarded straight to
    ``PeftModel.from_pretrained`` after parsing optional subfolder syntax.
    """

    name: str
    adapter_spec: str | None


def parse_variant(arg: str) -> VariantSpec:
    """Parse a --variant CLI value.

    Accepted forms:
        base                        -> name=base, adapter=None
        sft                         -> name=sft, adapter=None  (only when sft uses no adapter)
        sft:repo_id                 -> name=sft, adapter=repo_id
        sft:repo_id:subfolder       -> name=sft, adapter=repo_id:subfolder
        custom=path/or/repo         -> name=custom, adapter=path/or/repo
    """

    if "=" in arg:
        name, adapter = arg.split("=", 1)
        adapter = adapter.strip() or None
        if adapter and adapter.lower() == "base":
            adapter = None
        return VariantSpec(name=name.strip(), adapter_spec=adapter)
    if ":" in arg:
        name, _, adapter = arg.partition(":")
        return VariantSpec(name=name.strip(), adapter_spec=adapter.strip() or None)
    if arg.strip().lower() == "base":
        return VariantSpec(name="base", adapter_spec=None)
    return VariantSpec(name=arg.strip(), adapter_spec=None)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_base_model(base_model: str, *, dtype: str = "bfloat16"):
    """Lazy import: keeps tests fast on CPU-only hosts."""

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

    LOG.info("loading tokenizer + base model: %s (%s)", base_model, dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # We do batched inference via HFGeneratePolicy._generate which is
    # single-prompt at a time, but we still set left padding so any future
    # group-batched generate path works without surprise.
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
    )
    model.eval()
    return model, tokenizer


def _attach_adapter(model, adapter_spec: str):
    """Attach a PEFT LoRA adapter to ``model``.

    ``adapter_spec`` accepts ``<repo_id>:<subfolder>`` or a plain
    repo id / local path, mirroring the convention used by the GRPO
    trainer's ``sft_checkpoint`` field.
    """

    from peft import PeftModel

    if ":" in adapter_spec and not Path(adapter_spec).exists():
        repo_id, subfolder = adapter_spec.split(":", 1)
        load_target, load_kwargs = repo_id, {"subfolder": subfolder}
    else:
        local = Path(adapter_spec)
        if local.exists():
            load_target, load_kwargs = str(local), {}
        else:
            load_target, load_kwargs = adapter_spec, {}
    LOG.info("attaching PEFT adapter: %s (kwargs=%s)", load_target, load_kwargs)
    return PeftModel.from_pretrained(
        model,
        load_target,
        is_trainable=False,
        torch_device="cpu",
        **load_kwargs,
    )


def _move_to_cuda(model):
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        return model.to(device), device
    LOG.warning("CUDA not available — running eval on CPU (slow).")
    return model, torch.device("cpu")


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------


@dataclass
class RolloutRow:
    variant: str
    task_id: str
    seed: int
    success: bool
    total_reward: float
    process_reward: float
    terminal_reward: float
    steps: int
    parse_errors: int
    failure_mode: str = "unknown"
    reward_components: dict[str, float] = field(default_factory=dict)


def _run_one(
    *,
    variant: str,
    task_id: str,
    seed: int,
    model,
    tokenizer,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_history_steps: int,
    max_episode_steps: int,
    enable_thinking: bool,
) -> RolloutRow:
    from dronecaptureops.agent.hf_generate_policy import HFGeneratePolicy
    from dronecaptureops.agent.rollout import RolloutRunner
    from dronecaptureops.core.environment import DroneCaptureOpsEnvironment

    env = DroneCaptureOpsEnvironment()
    runner = RolloutRunner(env=env)
    policy = HFGeneratePolicy(
        model=model,
        tokenizer=tokenizer,
        env=env,
        task_id=task_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_history_steps=max_history_steps,
        enable_thinking=enable_thinking,
    )
    result = runner.run(policy, seed=seed, task_id=task_id, max_steps=max_episode_steps)

    parse_errors = sum(1 for s in result.trajectory if s.parse_error is not None)
    breakdown = result.reward_breakdown or {}
    components = {
        "process_reward": float(breakdown.get("process_reward", 0.0)),
        "evidence_success": float(breakdown.get("evidence_success", 0.0)),
        "required_coverage": float(breakdown.get("required_coverage", 0.0)),
        "issue_capture": float(breakdown.get("issue_capture", 0.0)),
        "operational_efficiency": float(breakdown.get("operational_efficiency", 0.0)),
        "grounded_report": float(breakdown.get("grounded_report", 0.0)),
        "safety_gate": float(breakdown.get("safety_gate", 1.0)),
        "integrity_gate": float(breakdown.get("integrity_gate", 1.0)),
    }

    # Compute a coarse failure mode using the existing eval_metrics helper.
    failure_mode = "unknown"
    try:
        from dronecaptureops.agent.eval_metrics import trajectory_metrics

        failure_mode = trajectory_metrics(result).failure_mode
    except Exception:  # noqa: BLE001 — never let metrics crash eval
        pass

    return RolloutRow(
        variant=variant,
        task_id=task_id,
        seed=seed,
        success=bool(result.success),
        total_reward=float(result.total_reward),
        process_reward=components["process_reward"],
        terminal_reward=float(result.total_reward) - components["process_reward"],
        steps=int(result.steps),
        parse_errors=parse_errors,
        failure_mode=failure_mode,
        reward_components=components,
    )


def _summarize_rows(rows: list[RolloutRow]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    n = len(rows)
    by_task: dict[str, dict[str, Any]] = {}
    for task_id in sorted({r.task_id for r in rows}):
        rs = [r for r in rows if r.task_id == task_id]
        by_task[task_id] = {
            "n": len(rs),
            "success_rate": sum(1.0 for r in rs if r.success) / len(rs),
            "mean_total_reward": statistics.mean(r.total_reward for r in rs),
            "mean_process_reward": statistics.mean(r.process_reward for r in rs),
            "mean_steps": statistics.mean(r.steps for r in rs),
            "mean_parse_errors": statistics.mean(r.parse_errors for r in rs),
        }
    failure_counts: dict[str, int] = {}
    for r in rows:
        failure_counts[r.failure_mode] = failure_counts.get(r.failure_mode, 0) + 1
    return {
        "n": n,
        "success_rate": sum(1.0 for r in rows if r.success) / n,
        "mean_total_reward": statistics.mean(r.total_reward for r in rows),
        "mean_process_reward": statistics.mean(r.process_reward for r in rows),
        "mean_terminal_reward": statistics.mean(r.terminal_reward for r in rows),
        "mean_steps": statistics.mean(r.steps for r in rows),
        "parse_error_rate": (
            sum(r.parse_errors for r in rows) / max(sum(r.steps for r in rows), 1)
        ),
        "by_task": by_task,
        "failure_modes": {k: v / n for k, v in failure_counts.items()},
    }


def evaluate_variant(
    *,
    variant: VariantSpec,
    base_model: str,
    held_out_tasks: list[str],
    seeds_per_task: int,
    seed_base: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_history_steps: int,
    max_episode_steps: int,
    enable_thinking: bool,
    dtype: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Load the model+adapter and score every (task, seed) cell."""

    LOG.info("=== variant: %s (adapter=%s) ===", variant.name, variant.adapter_spec or "(none)")
    model, tokenizer = _load_base_model(base_model, dtype=dtype)
    if variant.adapter_spec:
        model = _attach_adapter(model, variant.adapter_spec)
    model, _device = _move_to_cuda(model)
    model.eval()

    rows: list[RolloutRow] = []
    rows_path = output_dir / f"rollouts_{variant.name}.jsonl"
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        with rows_path.open("w") as handle:
            for ti, task_id in enumerate(held_out_tasks):
                for si in range(seeds_per_task):
                    seed = seed_base + 1000 * ti + si
                    LOG.info(
                        "  [%s] %s seed=%d (cell %d/%d)",
                        variant.name, task_id, seed,
                        ti * seeds_per_task + si + 1,
                        len(held_out_tasks) * seeds_per_task,
                    )
                    try:
                        row = _run_one(
                            variant=variant.name,
                            task_id=task_id,
                            seed=seed,
                            model=model,
                            tokenizer=tokenizer,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            max_history_steps=max_history_steps,
                            max_episode_steps=max_episode_steps,
                            enable_thinking=enable_thinking,
                        )
                    except Exception as exc:  # noqa: BLE001 — record + continue
                        LOG.exception("[%s] %s seed=%d crashed: %s", variant.name, task_id, seed, exc)
                        continue
                    rows.append(row)
                    handle.write(json.dumps(row.__dict__, sort_keys=True) + "\n")
                    handle.flush()
    finally:
        # Free GPU memory between variants — caller reloads base from scratch.
        try:
            del model
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    elapsed = time.time() - started
    summary = _summarize_rows(rows)
    summary.update({
        "variant": variant.name,
        "adapter_spec": variant.adapter_spec,
        "base_model": base_model,
        "elapsed_secs": elapsed,
        "seeds_per_task": seeds_per_task,
        "max_episode_steps": max_episode_steps,
    })
    return summary


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------


def _format_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def build_comparison_markdown(per_variant: list[dict[str, Any]]) -> str:
    """Render a comparison table across variants."""

    if not per_variant:
        return "# Eval comparison\n\nNo variants completed.\n"

    lines: list[str] = []
    lines.append("# Eval comparison: base vs SFT vs GRPO")
    lines.append("")
    lines.append("## Headline (held-out task suite)")
    lines.append("")
    lines.append("| variant | n | success | mean_reward | mean_proc | mean_steps | parse_err |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for v in per_variant:
        lines.append(
            f"| {v.get('variant', '?')} | {v.get('n', 0)} | "
            f"{_format_pct(v.get('success_rate', 0.0))} | "
            f"{v.get('mean_total_reward', 0.0):.4f} | "
            f"{v.get('mean_process_reward', 0.0):.4f} | "
            f"{v.get('mean_steps', 0.0):.1f} | "
            f"{_format_pct(v.get('parse_error_rate', 0.0))} |"
        )

    # Per-task breakdown.
    lines.append("")
    lines.append("## Per-task mean reward")
    lines.append("")
    task_ids = sorted({tid for v in per_variant for tid in v.get("by_task", {})})
    header = "| task | " + " | ".join(v.get("variant", "?") for v in per_variant) + " |"
    sep = "| --- |" + " ---: |" * len(per_variant)
    lines.append(header)
    lines.append(sep)
    for tid in task_ids:
        cells: list[str] = []
        for v in per_variant:
            cell = v.get("by_task", {}).get(tid)
            if cell is None:
                cells.append("—")
            else:
                cells.append(f"{cell.get('mean_total_reward', 0.0):.4f}")
        lines.append(f"| {tid} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Per-task success rate")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for tid in task_ids:
        cells = []
        for v in per_variant:
            cell = v.get("by_task", {}).get(tid)
            if cell is None:
                cells.append("—")
            else:
                cells.append(_format_pct(cell.get("success_rate", 0.0)))
        lines.append(f"| {tid} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Failure-mode distribution")
    lines.append("")
    all_modes = sorted({m for v in per_variant for m in v.get("failure_modes", {})})
    lines.append("| failure_mode | " + " | ".join(v.get("variant", "?") for v in per_variant) + " |")
    lines.append("| --- |" + " ---: |" * len(per_variant))
    for mode in all_modes:
        cells = [_format_pct(v.get("failure_modes", {}).get(mode, 0.0)) for v in per_variant]
        lines.append(f"| {mode} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    for v in per_variant:
        lines.append(
            f"- **{v.get('variant', '?')}** — base={v.get('base_model', '?')}, "
            f"adapter={v.get('adapter_spec') or '(none)'}, "
            f"seeds_per_task={v.get('seeds_per_task', '?')}, "
            f"max_episode_steps={v.get('max_episode_steps', '?')}, "
            f"elapsed={v.get('elapsed_secs', 0.0):.1f}s"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate base / SFT / GRPO variants on the held-out suite via HF generate."
    )
    parser.add_argument(
        "--base-model", required=True,
        help="HF model id of the base (LoRA adapters are stacked on top).",
    )
    parser.add_argument(
        "--variant", action="append", required=True,
        help=(
            "Variant spec. Repeatable. Forms: 'base', 'name=adapter_spec', "
            "'name:repo_id', 'name:repo_id:subfolder'. Use 'base' alone for the "
            "no-adapter run."
        ),
    )
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Override held-out task list (default: 7 standard ones).")
    parser.add_argument("--seeds-per-task", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=7777)
    parser.add_argument("--max-episode-steps", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Per-turn generate cap. Match the GRPO training value for like-for-like comparisons.")
    parser.add_argument("--max-history-steps", type=int, default=4,
                        help="Trim the chat history to the system prompt + first user + last K user/assistant pairs.")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Lower than rollout temp (0.9) — eval is greedier so noise across seeds shrinks.")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable Qwen <think> blocks during generation. Off by default (matches GRPO training).")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory where per-variant rollouts.jsonl, summary.json, and the comparison MD are written.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args(argv)

    held_out = list(args.tasks) if args.tasks else list(DEFAULT_HELD_OUT)
    variants = [parse_variant(v) for v in args.variant]
    if not variants:
        raise SystemExit("at least one --variant is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_variant: list[dict[str, Any]] = []

    for variant in variants:
        summary = evaluate_variant(
            variant=variant,
            base_model=args.base_model,
            held_out_tasks=held_out,
            seeds_per_task=args.seeds_per_task,
            seed_base=args.seed_base,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            max_history_steps=args.max_history_steps,
            max_episode_steps=args.max_episode_steps,
            enable_thinking=args.enable_thinking,
            dtype=args.dtype,
            output_dir=args.output_dir,
        )
        per_variant.append(summary)
        # Flush per-variant summary on the way so a mid-run crash still
        # leaves usable artefacts on disk.
        (args.output_dir / f"summary_{variant.name}.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )

    comparison = {
        "base_model": args.base_model,
        "tasks": held_out,
        "seeds_per_task": args.seeds_per_task,
        "max_episode_steps": args.max_episode_steps,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "variants": per_variant,
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True)
    )
    md = build_comparison_markdown(per_variant)
    (args.output_dir / "comparison.md").write_text(md)
    print(md)
    LOG.info("wrote eval artefacts under %s", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
