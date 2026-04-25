"""Held-out task evaluation for trained PPO checkpoints.

Loads a checkpoint (LoRA adapter + value head) and runs the existing
`RolloutRunner` against held-out tasks via vLLM. Emits a metrics JSON
suitable for both async-during-training eval and final report-time
benchmark runs.

Replaces the prior stub which just dispatched to `run_scripted_agent`.

CLI:
    python -m training.eval_policy --checkpoint artifacts/ppo-checkpoints/step_100 \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --output artifacts/eval/step_100.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dronecaptureops.agent.rollout import RolloutRunner
from dronecaptureops.agent.vllm_policy import VLLMEngine, VLLMPolicy
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


LOG = logging.getLogger("dronecaptureops.eval")


# Default held-out tasks — must stay in sync with sft_default.yaml.
DEFAULT_HELD_OUT = [
    "scheduled_crane_window_wait_or_detour",
    "honest_partial_report_open_items",
    "strict_severity_weighted_triage",
    "edge_row_quality_bar",
    "privacy_safe_alternate_evidence",
    "return_margin_decision_point",
    "route_replan_when_primary_viewpoint_blocked",
]


def _eval_one(
    task_id: str,
    seed: int,
    *,
    engine: VLLMEngine,
    lora_request: Any | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_episode_steps: int,
) -> dict[str, Any]:
    env = DroneCaptureOpsEnvironment()
    runner = RolloutRunner(env=env)
    policy = VLLMPolicy(
        engine=engine,
        env=env,
        task_id=task_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        lora_request=lora_request,
    )
    result = runner.run(policy, seed=seed, task_id=task_id, max_steps=max_episode_steps)
    parse_errors = sum(1 for s in result.trajectory if s.parse_error is not None)
    breakdown = result.reward_breakdown or {}
    return {
        "task_id": task_id,
        "seed": seed,
        "success": bool(result.success),
        "total_reward": float(result.total_reward),
        "steps": int(result.steps),
        "parse_errors": int(parse_errors),
        "process_reward": float(breakdown.get("process_reward", 0.0)),
        "evidence_success": float(breakdown.get("evidence_success", 0.0)),
        "required_coverage": float(breakdown.get("required_coverage", 0.0)),
        "issue_capture": float(breakdown.get("issue_capture", 0.0)),
        "operational_efficiency": float(breakdown.get("operational_efficiency", 0.0)),
        "grounded_report": float(breakdown.get("grounded_report", 0.0)),
        "safety_gate": float(breakdown.get("safety_gate", 1.0)),
        "integrity_gate": float(breakdown.get("integrity_gate", 1.0)),
    }


def evaluate_checkpoint(
    *,
    base_model: str,
    adapter_path: str | None,
    held_out_tasks: list[str],
    seeds_per_task: int,
    max_workers: int = 8,
    max_episode_steps: int = 40,
    temperature: float = 0.4,      # lower than rollout temp — eval is greedier
    top_p: float = 0.9,
    max_tokens: int = 1024,
    seed_base: int = 7777,
    vllm_max_model_len: int = 32768,
    vllm_gpu_memory_utilization: float = 0.85,
    vllm_dtype: str = "bfloat16",
) -> dict[str, Any]:
    """Run the eval matrix and return aggregated + per-rollout metrics."""

    # Build LoRARequest if an adapter is provided.
    lora_request = None
    enable_lora = adapter_path is not None
    if enable_lora:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("eval_policy", 1, str(adapter_path))

    LOG.info(
        "eval: base=%s adapter=%s tasks=%d seeds/task=%d",
        base_model, adapter_path or "(base)", len(held_out_tasks), seeds_per_task,
    )

    engine = VLLMEngine(
        model=base_model,
        max_model_len=vllm_max_model_len,
        dtype=vllm_dtype,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        enforce_eager=False,
        tensor_parallel_size=1,
        enable_lora=enable_lora,
    )

    # Build the (task_id, seed) matrix.
    pairs: list[tuple[str, int]] = []
    for ti, task_id in enumerate(held_out_tasks):
        for si in range(seeds_per_task):
            pairs.append((task_id, seed_base + 1000 * ti + si))

    t0 = time.perf_counter()
    rollouts: list[dict[str, Any]] = []
    workers = max(1, min(max_workers, len(pairs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _eval_one,
                task_id,
                seed,
                engine=engine,
                lora_request=lora_request,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_episode_steps=max_episode_steps,
            )
            for task_id, seed in pairs
        ]
        for fut in as_completed(futures):
            rollouts.append(fut.result())

    elapsed = time.perf_counter() - t0

    # Aggregations
    n = len(rollouts) or 1
    success_rate = sum(1 for r in rollouts if r["success"]) / n
    parse_error_rate = sum(r["parse_errors"] for r in rollouts) / max(
        sum(r["steps"] for r in rollouts), 1
    )
    mean_terminal = statistics.mean(r["total_reward"] for r in rollouts) if rollouts else 0.0
    mean_process = statistics.mean(r["process_reward"] for r in rollouts) if rollouts else 0.0
    process_to_terminal = (
        mean_process / mean_terminal if abs(mean_terminal) > 1e-6 else float("inf")
    )

    by_task: dict[str, dict[str, Any]] = {}
    for task_id in sorted({r["task_id"] for r in rollouts}):
        rs = [r for r in rollouts if r["task_id"] == task_id]
        by_task[task_id] = {
            "n": len(rs),
            "success_rate": sum(1 for r in rs if r["success"]) / len(rs),
            "mean_total_reward": statistics.mean(r["total_reward"] for r in rs),
            "mean_steps": statistics.mean(r["steps"] for r in rs),
            "mean_evidence_success": statistics.mean(r["evidence_success"] for r in rs),
        }

    return {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "n_rollouts": len(rollouts),
        "elapsed_secs": elapsed,
        "success_rate": success_rate,
        "mean_terminal_reward": mean_terminal,
        "mean_process_reward": mean_process,
        "process_to_terminal_ratio": process_to_terminal,
        "parse_error_rate": parse_error_rate,
        "by_task": by_task,
        "rollouts": rollouts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO/SFT checkpoint on held-out tasks.")
    parser.add_argument("--base-model", required=True, help="Base HF model id (must match training).")
    parser.add_argument("--adapter", "--checkpoint", dest="adapter", default=None,
                        help="Path to a LoRA adapter directory (omit to eval base model).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Where to write metrics JSON (default: stdout only).")
    parser.add_argument("--seeds-per-task", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-episode-steps", type=int, default=40)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Override held-out task list (default: 7 standard ones).")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()

    held_out = list(args.tasks) if args.tasks else DEFAULT_HELD_OUT
    metrics = evaluate_checkpoint(
        base_model=args.base_model,
        adapter_path=args.adapter,
        held_out_tasks=held_out,
        seeds_per_task=args.seeds_per_task,
        max_workers=args.max_workers,
        max_episode_steps=args.max_episode_steps,
        temperature=args.temperature,
        vllm_gpu_memory_utilization=args.gpu_memory_utilization,
    )

    summary = {k: v for k, v in metrics.items() if k != "rollouts"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True))
        LOG.info("wrote eval metrics to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
