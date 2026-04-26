"""Evaluate a PPO-trained LoRA adapter on the held-out task suite.

Replicates the SFT eval matrix EXACTLY (same 7 tasks × 2 seeds = 14 rollouts,
same seeds per task) so PPO results can be diffed directly against
`eval_outputs/base_vs_sft_report/raw/summary_sft.json` and `rollouts_sft.jsonl`.

Loads the base model + PPO LoRA adapter via PEFT and runs rollouts through the
in-process HFLocalEngine (the same one PPO training used — no vLLM, no IPC bug
class). Writes rollouts JSONL + summary JSON in the exact same format as the
existing eval reports.

Usage:
    python -m training.eval_ppo_adapter \\
        --base-model Qwen/Qwen3-4B-Instruct-2507 \\
        --adapter adityabhaskara/dronecaptureops-ppo-qwen3-4b-hf-local-Ashort2:output/step_10/adapter \\
        --variant ppo_step10 \\
        --output-dir artifacts/eval/ppo_step10
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from dronecaptureops.agent.eval_metrics import classify_failure_mode
from dronecaptureops.agent.hf_local_engine import HFLocalEngine, HFLocalPolicy
from dronecaptureops.agent.rollout import RolloutRunner
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


LOG = logging.getLogger("dronecaptureops.eval.ppo_adapter")


# Replicates the (task, seed) matrix from
# eval_outputs/base_vs_sft_report/raw/rollouts_sft.jsonl exactly so PPO eval is
# directly diff-able against the SFT and base eval numbers.
EVAL_MATRIX: list[tuple[str, int]] = [
    ("edge_row_quality_bar", 10777),
    ("edge_row_quality_bar", 10778),
    ("honest_partial_report_open_items", 8777),
    ("honest_partial_report_open_items", 8778),
    ("privacy_safe_alternate_evidence", 11777),
    ("privacy_safe_alternate_evidence", 11778),
    ("return_margin_decision_point", 12777),
    ("return_margin_decision_point", 12778),
    ("route_replan_when_primary_viewpoint_blocked", 13777),
    ("route_replan_when_primary_viewpoint_blocked", 13778),
    ("scheduled_crane_window_wait_or_detour", 7777),
    ("scheduled_crane_window_wait_or_detour", 7778),
    ("strict_severity_weighted_triage", 9777),
    ("strict_severity_weighted_triage", 9778),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a PPO LoRA adapter against the held-out suite.")
    p.add_argument("--base-model", required=True, help="HF Hub model id of the base.")
    p.add_argument(
        "--adapter",
        required=True,
        help="Adapter spec: local path OR `repo_id:subfolder` (e.g., user/repo:output/step_10/adapter).",
    )
    p.add_argument("--variant", required=True, help="Label written into each rollout row, e.g. ppo_step10.")
    p.add_argument("--output-dir", required=True, type=Path, help="Where rollouts.jsonl + summary.json land.")
    p.add_argument("--max-episode-steps", type=int, default=20)
    p.add_argument("--max-history-steps", type=int, default=12)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.4, help="Eval temperature (lower than training's 0.7).")
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def _resolve_adapter(spec: str) -> tuple[str, dict[str, Any]]:
    """Accept `local/path` or `repo_id:subfolder` style.

    Returns (load_target, peft_kwargs) suitable for `PeftModel.from_pretrained`.
    """

    if ":" in spec and not Path(spec).exists():
        repo_id, subfolder = spec.split(":", 1)
        return repo_id, {"subfolder": subfolder}
    return spec, {}


def _load_model(base_model: str, adapter_spec: str, dtype: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    LOG.info("loading base model %s (%s)", base_model, dtype)
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch_dtype,
        device_map="cuda",
    )
    load_target, kwargs = _resolve_adapter(adapter_spec)
    LOG.info("loading PPO LoRA adapter %s (kwargs=%s)", load_target, kwargs)
    model = PeftModel.from_pretrained(base, load_target, **kwargs)
    model.eval()
    return model, tokenizer


def _row_from_result(result, *, task_id: str, seed: int, variant: str) -> dict[str, Any]:
    """Same row schema as eval_outputs/base_vs_sft_report/raw/rollouts_sft.jsonl."""

    parse_errors = sum(1 for s in result.trajectory if s.parse_error is not None)
    terminal_reward = float(result.trajectory[-1].reward) if result.trajectory else 0.0
    process_reward = float(result.total_reward) - terminal_reward
    # reward_breakdown stored on the result is a dict; pick the components if present
    rb = result.reward_breakdown or {}
    components = rb.get("components") if isinstance(rb, dict) else None
    if components is None and isinstance(rb, dict):
        # Some breakdowns store the per-component scalars at the top level
        components = {k: float(v) for k, v in rb.items() if isinstance(v, (int, float))}
    components = {k: float(v) for k, v in (components or {}).items()}
    return {
        "failure_mode": classify_failure_mode(result),
        "parse_errors": int(parse_errors),
        "process_reward": float(process_reward),
        "reward_components": components,
        "seed": int(seed),
        "steps": int(result.steps),
        "success": bool(result.success),
        "task_id": task_id,
        "terminal_reward": float(terminal_reward),
        "total_reward": float(result.total_reward),
        "variant": variant,
    }


def _summary(rows: list[dict[str, Any]], *, adapter_spec: str, base_model: str) -> dict[str, Any]:
    """Same summary schema as eval_outputs/base_vs_sft_report/raw/summary_sft.json."""

    by_task: dict[str, dict[str, Any]] = {}
    tasks = sorted({r["task_id"] for r in rows})
    for t in tasks:
        sub = [r for r in rows if r["task_id"] == t]
        n = len(sub)
        by_task[t] = {
            "mean_parse_errors": round(sum(r["parse_errors"] for r in sub) / n, 4),
            "mean_process_reward": round(sum(r["process_reward"] for r in sub) / n, 4),
            "mean_steps": round(sum(r["steps"] for r in sub) / n, 4),
            "mean_total_reward": round(sum(r["total_reward"] for r in sub) / n, 4),
            "n": n,
            "success_rate": round(sum(1 for r in sub if r["success"]) / n, 4),
        }
    n = len(rows)
    overall = {
        "mean_total_reward": round(sum(r["total_reward"] for r in rows) / n, 4) if n else 0.0,
        "mean_terminal_reward": round(sum(r["terminal_reward"] for r in rows) / n, 4) if n else 0.0,
        "mean_process_reward": round(sum(r["process_reward"] for r in rows) / n, 4) if n else 0.0,
        "mean_steps": round(sum(r["steps"] for r in rows) / n, 4) if n else 0.0,
        "success_rate": round(sum(1 for r in rows if r["success"]) / n, 4) if n else 0.0,
        "parse_error_rate": round(sum(r["parse_errors"] for r in rows) / n, 4) if n else 0.0,
        "n_rollouts": n,
    }
    return {
        "adapter_spec": adapter_spec,
        "base_model": base_model,
        "by_task": by_task,
        "overall": overall,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = args.output_dir / "rollouts.jsonl"
    summary_path = args.output_dir / "summary.json"

    model, tokenizer = _load_model(args.base_model, args.adapter, args.dtype)
    import torch

    device = torch.device("cuda")
    engine = HFLocalEngine(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_batch_size=4,                # eval is sequential per (task, seed)
        max_history_steps=args.max_history_steps,
    )

    rows: list[dict[str, Any]] = []
    LOG.info("running %d rollouts for variant=%s", len(EVAL_MATRIX), args.variant)
    t0 = time.perf_counter()
    for i, (task_id, seed) in enumerate(EVAL_MATRIX, start=1):
        env = DroneCaptureOpsEnvironment()
        runner = RolloutRunner(env=env)
        policy = HFLocalPolicy(
            engine=engine,
            env=env,
            task_id=task_id,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_new_tokens,
            max_history_steps=args.max_history_steps,
        )
        t_cell = time.perf_counter()
        try:
            result = runner.run(policy, seed=seed, task_id=task_id, max_steps=args.max_episode_steps)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("rollout failed: task=%s seed=%s", task_id, seed)
            # Record a zero-reward failure row so the matrix is complete
            rows.append({
                "failure_mode": "error",
                "parse_errors": 0,
                "process_reward": 0.0,
                "reward_components": {},
                "seed": int(seed),
                "steps": 0,
                "success": False,
                "task_id": task_id,
                "terminal_reward": 0.0,
                "total_reward": 0.0,
                "variant": args.variant,
            })
            continue
        row = _row_from_result(result, task_id=task_id, seed=seed, variant=args.variant)
        rows.append(row)
        with rollouts_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        LOG.info(
            "[%d/%d] %s seed=%s reward=%.3f success=%s steps=%d failure=%s (%.1fs)",
            i, len(EVAL_MATRIX), task_id, seed,
            row["total_reward"], row["success"], row["steps"], row["failure_mode"],
            time.perf_counter() - t_cell,
        )

    summary = _summary(rows, adapter_spec=args.adapter, base_model=args.base_model)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    LOG.info("wrote %s and %s", rollouts_path, summary_path)
    LOG.info(
        "DONE — %d rollouts in %.1fs. Overall: reward=%.3f success=%.2f%% parse_err=%.2f%%",
        len(rows), time.perf_counter() - t0,
        summary["overall"]["mean_total_reward"],
        summary["overall"]["success_rate"] * 100,
        summary["overall"]["parse_error_rate"] * 100,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
