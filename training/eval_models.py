"""Cold-base evaluation harness: run any model on (task × seed) matrices.

Loads each model once via vLLM (so we don't pay 1-2 min startup per
episode), then drives the agent harness through the standard
`RolloutRunner`. The output is per-row JSONL plus an aggregated table.

Usage:
    # short horizon — current 15-task suite plus any auto-included
    # short-horizon tasks
    python -m training.eval_models \\
        --models "Qwen/Qwen3-14B,Qwen/Qwen3-30B-A3B-Instruct-2507,Qwen/Qwen3-32B" \\
        --task-tier short \\
        --seeds 3 \\
        --output artifacts/eval/short.jsonl

    # stretch horizon — multi-block tasks
    python -m training.eval_models \\
        --models "Qwen/Qwen3-14B,Qwen/Qwen3-30B-A3B-Instruct-2507" \\
        --task-tier stretch \\
        --seeds 3 \\
        --output artifacts/eval/stretch.jsonl

The HF_TOKEN environment variable is read by vLLM/transformers. Gated
models load fine once it's set.

Notes:
- Model loading is lazy. Tests in this file run without vllm installed.
- Each (model, task, seed) cell is sequential within a process. For
  parallel evaluation across cells, run multiple processes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from dronecaptureops.agent import RolloutResult, RolloutRunner, TaskOraclePolicy
from dronecaptureops.agent.eval_metrics import (
    CHECKPOINT_NAMES,
    FAILURE_MODES,
    aggregate_diagnostics,
    trajectory_metrics,
)
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("dronecaptureops.eval.models")


# Tier definitions are based on task max_steps. New tasks added to the
# catalog auto-route to the right tier without code changes.
SHORT_TIER_MAX_STEPS = 50
STRETCH_TIER_MIN_STEPS = 51


# ---------------------------------------------------------------------------
# Plan resolution
# ---------------------------------------------------------------------------


def resolve_tasks(*, tier: str, explicit: list[str] | None) -> list[str]:
    """Return the task IDs to evaluate, given a tier filter or explicit list."""

    if explicit:
        unknown = sorted(set(explicit) - set(SOLAR_TASKS))
        if unknown:
            raise SystemExit(f"unknown tasks: {unknown}")
        return list(explicit)

    if tier == "short":
        return [tid for tid, spec in SOLAR_TASKS.items() if spec.max_steps <= SHORT_TIER_MAX_STEPS]
    if tier == "stretch":
        return [tid for tid, spec in SOLAR_TASKS.items() if spec.max_steps >= STRETCH_TIER_MIN_STEPS]
    if tier == "all":
        return list(SOLAR_TASKS.keys())
    raise SystemExit(f"unknown task tier: {tier!r} (expected short/stretch/all)")


@dataclass
class EvalCell:
    """One (model, task, seed) assignment."""

    model: str
    task_id: str
    seed: int
    max_steps: int


def build_plan(*, models: list[str], task_ids: list[str], seeds: int, seed_offset: int) -> list[EvalCell]:
    cells: list[EvalCell] = []
    for model in models:
        for task_id in task_ids:
            spec = SOLAR_TASKS[task_id]
            for s in range(seeds):
                cells.append(
                    EvalCell(
                        model=model,
                        task_id=task_id,
                        seed=seed_offset + s,
                        max_steps=spec.max_steps,
                    )
                )
    return cells


# ---------------------------------------------------------------------------
# vLLM glue
# ---------------------------------------------------------------------------


def _load_engine(
    model: str,
    *,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    enforce_eager: bool,
    trust_remote_code: bool,
):
    """Lazy-imported vLLM engine factory."""

    from dronecaptureops.agent import VLLMEngine  # type: ignore[attr-defined]

    return VLLMEngine(
        model=model,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
        enable_thinking=False,
    )


def _make_policy(engine, env: DroneCaptureOpsEnvironment, *, task_id: str, temperature: float, max_tokens: int, max_history_steps: int):
    from dronecaptureops.agent import VLLMPolicy  # type: ignore[attr-defined]

    return VLLMPolicy(
        engine=engine,
        env=env,
        task_id=task_id,
        temperature=temperature,
        max_tokens=max_tokens,
        max_history_steps=max_history_steps,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class EvalRow:
    model: str
    task_id: str
    seed: int
    success: bool
    complete: bool
    total_reward: float
    steps: int
    parse_error_count: int
    safety_violations: list[str] = field(default_factory=list)
    final_status: dict[str, Any] = field(default_factory=dict)
    # Diagnostic profile populated by trajectory_metrics():
    failure_mode: str = "unknown"
    checkpoints: dict[str, bool] = field(default_factory=dict)
    tool_calls: dict[str, int] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    reward_components: dict[str, float] = field(default_factory=dict)
    oracle_comparison: dict[str, Any] = field(default_factory=dict)


def _row_from_result(
    model: str,
    cell: EvalCell,
    result: RolloutResult,
    *,
    oracle_result: RolloutResult | None = None,
) -> EvalRow:
    final = result.final_observation
    checklist = final.get("checklist_status", {})
    parse_errors = sum(1 for step in result.trajectory if step.parse_error)
    safety_violations = [
        warning for warning in final.get("warnings", [])
        if "violation" in str(warning) or "unsafe" in str(warning)
    ]
    metrics = trajectory_metrics(result, oracle_result=oracle_result)
    return EvalRow(
        model=model,
        task_id=cell.task_id,
        seed=cell.seed,
        success=bool(result.success),
        complete=bool(checklist.get("complete")),
        total_reward=round(result.total_reward, 4),
        steps=result.steps,
        parse_error_count=parse_errors,
        safety_violations=safety_violations,
        final_status=checklist,
        failure_mode=metrics.failure_mode,
        checkpoints=metrics.checkpoints,
        tool_calls=metrics.tool_calls,
        coverage=metrics.coverage,
        safety=metrics.safety,
        reward_components=metrics.reward_components,
        oracle_comparison=metrics.oracle_comparison,
    )


def aggregate(rows: list[EvalRow]) -> dict[str, Any]:
    """Per-(model, task) summary + per-model summary + diagnostic profile."""

    by_model_task: dict[tuple[str, str], list[EvalRow]] = defaultdict(list)
    by_model: dict[str, list[EvalRow]] = defaultdict(list)
    for row in rows:
        by_model_task[(row.model, row.task_id)].append(row)
        by_model[row.model].append(row)

    per_cell = {}
    for (model, task_id), cell_rows in by_model_task.items():
        per_cell[f"{model}::{task_id}"] = {
            "model": model,
            "task_id": task_id,
            "n": len(cell_rows),
            "success_rate": mean(1.0 if r.success else 0.0 for r in cell_rows),
            "complete_rate": mean(1.0 if r.complete else 0.0 for r in cell_rows),
            "mean_reward": mean(r.total_reward for r in cell_rows),
            "mean_steps": mean(r.steps for r in cell_rows),
            "mean_parse_errors": mean(r.parse_error_count for r in cell_rows),
            "failure_modes": _failure_mode_breakdown(cell_rows),
        }

    per_model = {}
    for model, model_rows in by_model.items():
        per_model[model] = {
            "n": len(model_rows),
            "success_rate": mean(1.0 if r.success else 0.0 for r in model_rows),
            "complete_rate": mean(1.0 if r.complete else 0.0 for r in model_rows),
            "mean_reward": mean(r.total_reward for r in model_rows),
            "mean_steps": mean(r.steps for r in model_rows),
            "mean_parse_errors": mean(r.parse_error_count for r in model_rows),
        }

    diagnostics = aggregate_diagnostics([_row_to_jsonable(row) for row in rows])

    return {"per_cell": per_cell, "per_model": per_model, "diagnostics": diagnostics}


def format_summary(summary: dict[str, Any], rows: list[EvalRow]) -> str:
    """Render aggregated tables + diagnostic breakdowns for stdout."""

    lines: list[str] = []
    lines.append("\n=== per-model summary ===")
    lines.append(
        f"{'model':<48} {'n':>4} {'success':>8} {'complete':>9} {'mean_r':>7} {'mean_steps':>11} {'parse_err':>10}"
    )
    for model, agg in sorted(summary["per_model"].items()):
        lines.append(
            f"{model:<48} {agg['n']:>4} {agg['success_rate']:>8.2%} {agg['complete_rate']:>9.2%} "
            f"{agg['mean_reward']:>7.3f} {agg['mean_steps']:>11.1f} {agg['mean_parse_errors']:>10.2f}"
        )

    lines.append("\n=== failure-mode distribution per model ===")
    diagnostics = summary.get("diagnostics", {})
    for model, agg in sorted(diagnostics.items()):
        lines.append(f"\n  {model}:")
        for mode, frac in sorted(agg["failure_mode_distribution"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {mode:<24} {frac:>6.1%}")

    lines.append("\n=== capability checkpoint completion rate per model ===")
    for model, agg in sorted(diagnostics.items()):
        lines.append(f"\n  {model}:")
        for name in CHECKPOINT_NAMES:
            rate = agg["checkpoint_completion_rate"].get(name, 0.0)
            bar = "█" * int(rate * 20) + "·" * (20 - int(rate * 20))
            lines.append(f"    {name:<24} {rate:>6.1%}  {bar}")

    lines.append("\n=== mean tool calls per episode (top 10 per model) ===")
    for model, agg in sorted(diagnostics.items()):
        lines.append(f"\n  {model}:")
        top_tools = list(agg["tool_calls_per_episode"].items())[:10]
        for tool, mean_count in top_tools:
            lines.append(f"    {tool:<32} {mean_count:>6.2f}")

    lines.append("\n=== per-(model, task) ===")
    lines.append(
        f"{'model':<40} {'task_id':<28} {'n':>4} {'success':>8} {'complete':>9} {'mean_r':>7} {'mean_steps':>11}"
    )
    for key, agg in sorted(summary["per_cell"].items()):
        lines.append(
            f"{agg['model']:<40} {agg['task_id']:<28} {agg['n']:>4} "
            f"{agg['success_rate']:>8.2%} {agg['complete_rate']:>9.2%} "
            f"{agg['mean_reward']:>7.3f} {agg['mean_steps']:>11.1f}"
        )
    return "\n".join(lines)


def _failure_mode_breakdown(rows: list[EvalRow]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.failure_mode] = counts.get(row.failure_mode, 0) + 1
    n = len(rows)
    return {mode: round(counts[mode] / n, 4) for mode in counts}


def _precompute_oracle_refs(plan: list["EvalCell"]) -> dict[tuple[str, int], RolloutResult]:
    """Run TaskOraclePolicy once per unique (task_id, seed) for comparison.

    Cheap (no LLM); ~1s per task on CPU. Cached so each base-model rollout
    of the same (task, seed) reuses the same oracle reference.
    """

    refs: dict[tuple[str, int], RolloutResult] = {}
    seen: set[tuple[str, int]] = set()
    for cell in plan:
        key = (cell.task_id, cell.seed)
        if key in seen:
            continue
        seen.add(key)
        env = DroneCaptureOpsEnvironment()
        runner = RolloutRunner(env=env)
        try:
            refs[key] = runner.run(
                TaskOraclePolicy(task_id=cell.task_id),
                seed=cell.seed,
                task_id=cell.task_id,
                max_steps=cell.max_steps,
            )
        except Exception as exc:  # noqa: BLE001 — never let oracle failure crash eval
            LOG.warning("oracle reference failed for %s seed=%d: %s", cell.task_id, cell.seed, exc)
    return refs


def _row_to_jsonable(row: EvalRow) -> dict[str, Any]:
    return {
        "model": row.model,
        "task_id": row.task_id,
        "seed": row.seed,
        "success": row.success,
        "complete": row.complete,
        "total_reward": row.total_reward,
        "steps": row.steps,
        "parse_error_count": row.parse_error_count,
        "safety_violations": row.safety_violations,
        "final_status": row.final_status,
        "failure_mode": row.failure_mode,
        "checkpoints": row.checkpoints,
        "tool_calls": row.tool_calls,
        "coverage": row.coverage,
        "safety": row.safety,
        "reward_components": row.reward_components,
        "oracle_comparison": row.oracle_comparison,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base / fine-tuned models on DroneCaptureOps tasks via vLLM.")
    parser.add_argument("--models", required=True, help="Comma-separated HF model IDs.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task-tier", default="short", choices=["short", "stretch", "all"], help="Filter tasks by horizon tier.")
    group.add_argument("--tasks", default=None, help="Comma-separated task IDs (overrides --task-tier).")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="JSONL path for per-cell rows.")
    parser.add_argument("--summary-output", type=Path, default=None, help="Optional JSON path for aggregated summary.")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-history-steps", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs (slower; sometimes needed for compat).")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without launching any model.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    explicit_tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    task_ids = resolve_tasks(tier=args.task_tier, explicit=explicit_tasks)
    plan = build_plan(models=models, task_ids=task_ids, seeds=args.seeds, seed_offset=args.seed_offset)

    LOG.info("plan: %d cells across %d models, %d tasks, %d seeds", len(plan), len(models), len(task_ids), args.seeds)
    if args.dry_run:
        for cell in plan:
            print(f"  - {cell.model} :: {cell.task_id} :: seed={cell.seed} :: max_steps={cell.max_steps}")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[EvalRow] = []
    started = time.time()

    # Pre-compute oracle reference trajectories per (task_id, seed) — used
    # for step-efficiency and tool-overlap metrics. The oracle is fast
    # (~1s per task) and runs without any LLM, so this is cheap.
    LOG.info("computing oracle reference trajectories for %d tasks", len(set(c.task_id for c in plan)))
    oracle_refs = _precompute_oracle_refs(plan)

    # Group cells by model so we load each model exactly once.
    cells_by_model: dict[str, list[EvalCell]] = defaultdict(list)
    for cell in plan:
        cells_by_model[cell.model].append(cell)

    with args.output.open("w") as handle:
        for model, cells in cells_by_model.items():
            LOG.info("loading model %s (%d cells)", model, len(cells))
            engine = _load_engine(
                model,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                enforce_eager=args.enforce_eager,
                trust_remote_code=args.trust_remote_code,
            )
            try:
                for idx, cell in enumerate(cells, start=1):
                    LOG.info("[%s] cell %d/%d :: %s :: seed=%d", model, idx, len(cells), cell.task_id, cell.seed)
                    env = DroneCaptureOpsEnvironment()
                    policy = _make_policy(
                        engine, env,
                        task_id=cell.task_id,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        max_history_steps=args.max_history_steps,
                    )
                    runner = RolloutRunner(env=env)
                    result = runner.run(
                        policy,
                        seed=cell.seed,
                        task_id=cell.task_id,
                        max_steps=cell.max_steps,
                    )
                    oracle_ref = oracle_refs.get((cell.task_id, cell.seed))
                    row = _row_from_result(model, cell, result, oracle_result=oracle_ref)
                    rows.append(row)
                    handle.write(json.dumps(_row_to_jsonable(row), sort_keys=True) + "\n")
                    handle.flush()
            finally:
                # vLLM holds a process-wide CUDA context; release it before
                # loading the next model.
                del engine

    elapsed = time.time() - started
    LOG.info("done; %d cells in %.1fs (%.1fs/cell avg)", len(rows), elapsed, elapsed / max(len(rows), 1))

    summary = aggregate(rows)
    print(format_summary(summary, rows))

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True))
        LOG.info("summary written to %s", args.summary_output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
