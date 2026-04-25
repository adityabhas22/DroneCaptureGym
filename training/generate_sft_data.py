"""Generate SFT warm-start data for DroneCaptureOps tasks.

Driven by a YAML config so new tasks plug in without touching this file:
the script reads `dronecaptureops.tasks.solar_tasks.SOLAR_TASKS` at runtime
and applies the config's include/exclude/per-task overrides on top. As soon
as a new task is added to that catalog it becomes eligible for SFT data
unless explicitly excluded or held out.

Usage:
    python -m training.generate_sft_data
    python -m training.generate_sft_data \\
        --config training/configs/sft_default.yaml \\
        --output artifacts/sft/run-2026-04-25.jsonl

CLI flags override config values for the most common knobs (seeds, output,
include/exclude). Per-task overrides (max_steps, policies, etc.) only live
in YAML.

Output JSONL — one example per line, each with:
    task_id, seed, policy, reward, success, complete, steps, messages,
    reward_breakdown, held_out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from dronecaptureops.agent import (
    RandomPolicy,
    RolloutResult,
    RolloutRunner,
    ScriptedPolicy,
    SpecAwareScriptedPolicy,
    TaskOraclePolicy,
    trajectory_to_chat_messages,
)
from dronecaptureops.agent.policies import Policy
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "sft_default.yaml"

LOG = logging.getLogger("dronecaptureops.sft.generate")


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class PolicyEntry(BaseModel):
    """One policy in the data-generation rotation."""

    name: str = "task_oracle"
    weight: int = Field(default=1, ge=1)

    @classmethod
    def parse(cls, data: Any) -> "PolicyEntry":
        if isinstance(data, str):
            return cls(name=data)
        return cls.model_validate(data)


class TaskOverride(BaseModel):
    """Per-task override layered on top of the global SFTGenConfig."""

    task_id: str
    seeds: list[int] | None = None
    seeds_per_task: int | None = Field(default=None, ge=0)
    max_steps: int | None = Field(default=None, ge=1)
    policies: list[PolicyEntry] | None = None


class SFTGenConfig(BaseModel):
    """Top-level config schema for `generate_sft_data.py`."""

    output: str = "artifacts/sft/sft-warmstart.jsonl"
    seeds_per_task: int = Field(default=8, ge=1)
    max_steps: int = Field(default=40, ge=1)
    policies: list[PolicyEntry] = Field(default_factory=lambda: [PolicyEntry(name="task_oracle")])
    include_tasks: list[str] | None = None
    exclude_tasks: list[str] = Field(default_factory=list)
    tasks: list[TaskOverride] = Field(default_factory=list)
    require_success: bool = True
    deduplicate: bool = True
    use_tool_calls: bool = False
    shuffle: bool = True
    shuffle_seed: int = 1
    held_out_tasks: list[str] = Field(default_factory=list)


def load_config(path: Path | None) -> SFTGenConfig:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        LOG.warning("config path %s not found, using built-in defaults", path)
        return SFTGenConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    if "policies" in raw:
        raw["policies"] = [PolicyEntry.parse(entry).model_dump() for entry in raw["policies"]]
    if "tasks" in raw:
        normalised: list[dict[str, Any]] = []
        for entry in raw["tasks"]:
            entry = dict(entry)
            if "policies" in entry and entry["policies"] is not None:
                entry["policies"] = [PolicyEntry.parse(p).model_dump() for p in entry["policies"]]
            normalised.append(entry)
        raw["tasks"] = normalised
    try:
        return SFTGenConfig.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(f"invalid SFT generation config: {exc}") from exc


# ---------------------------------------------------------------------------
# Rollout dispatch
# ---------------------------------------------------------------------------


@dataclass
class TaskPlan:
    """Resolved per-task plan after applying overrides."""

    task_id: str
    seeds: list[int]
    max_steps: int
    policies: list[PolicyEntry]
    held_out: bool = False


def resolve_task_plans(config: SFTGenConfig) -> list[TaskPlan]:
    """Compute the actual (task, seeds, policies, max_steps) plans to run.

    Reads `SOLAR_TASKS` live so tasks added after this script was written
    show up automatically. Held-out tasks are still emitted as plans so
    the script can write them to a separate eval JSONL — but the writer
    skips them by default (see `--with-heldout`).
    """

    overrides_by_task = {entry.task_id: entry for entry in config.tasks}
    held_out = set(config.held_out_tasks)

    if config.include_tasks is not None:
        candidate_ids = list(config.include_tasks)
    else:
        candidate_ids = list(SOLAR_TASKS.keys())
    excluded = set(config.exclude_tasks)

    plans: list[TaskPlan] = []
    for task_id in candidate_ids:
        if task_id not in SOLAR_TASKS:
            LOG.warning("config references unknown task %r — skipping", task_id)
            continue
        if task_id in excluded:
            continue
        override = overrides_by_task.get(task_id)
        seeds_count = (override.seeds_per_task if override else None) or config.seeds_per_task
        seeds = (
            list(override.seeds)
            if override and override.seeds is not None
            else list(range(seeds_count))
        )
        max_steps = (override.max_steps if override else None) or config.max_steps
        policies = (
            list(override.policies)
            if override and override.policies is not None
            else list(config.policies)
        )
        plans.append(
            TaskPlan(
                task_id=task_id,
                seeds=seeds,
                max_steps=max_steps,
                policies=policies,
                held_out=task_id in held_out,
            )
        )
    return plans


def build_policy(name: str, *, task_id: str, seed: int) -> Policy:
    # `spec_aware_scripted_s{0,1,2}` selects between three valid solution
    # paths in the spec-aware solver — careful, streamlined, diagnostic.
    # Listing all three in the YAML's `policies:` rotation triples the
    # action-sequence diversity per task, breaking the task_id → fixed-
    # sequence shortcut the model would otherwise memorise.
    if name == "spec_aware_scripted" or name == "spec_aware_scripted_s0":
        return SpecAwareScriptedPolicy(task_id=task_id, seed=seed, strategy=0)
    if name == "spec_aware_scripted_s1":
        return SpecAwareScriptedPolicy(task_id=task_id, seed=seed, strategy=1)
    if name == "spec_aware_scripted_s2":
        return SpecAwareScriptedPolicy(task_id=task_id, seed=seed, strategy=2)
    if name == "spec_aware_scripted_s3":
        return SpecAwareScriptedPolicy(task_id=task_id, seed=seed, strategy=3)
    if name == "task_oracle":
        # Legacy oracle (only solves ~20 of 45 tasks). Kept for backward compat
        # with existing eval pipelines that pass it as a comparison baseline.
        return TaskOraclePolicy(task_id=task_id)
    if name == "scripted":
        return ScriptedPolicy()
    if name == "random":
        return RandomPolicy(seed=seed)
    raise SystemExit(f"unsupported policy in SFT config: {name!r}")


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclass
class GenerationStats:
    """Per-task accounting for end-of-run summary."""

    rollouts_attempted: int = 0
    rollouts_kept: int = 0
    rollouts_failed_success: int = 0
    rollouts_dropped_duplicate: int = 0
    avg_reward: float = 0.0
    _reward_sum: float = field(default=0.0, repr=False)


def _hash_messages(messages: list[dict[str, Any]]) -> str:
    payload = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _record_for(result: RolloutResult, *, plan: TaskPlan, policy_name: str, use_tool_calls: bool) -> dict[str, Any]:
    final_status = result.final_observation.get("checklist_status", {})
    messages = trajectory_to_chat_messages(result, use_tool_calls=use_tool_calls)
    return {
        "task_id": result.task_id or plan.task_id,
        "seed": result.seed,
        "policy": policy_name,
        "reward": round(result.total_reward, 4),
        "success": bool(result.success),
        "complete": bool(final_status.get("complete")),
        "steps": result.steps,
        "anomalies_detected": final_status.get("anomalies_detected") or [],
        "anomaly_rgb_pairs": final_status.get("anomaly_rgb_pairs") or {},
        "reward_breakdown": result.reward_breakdown,
        "messages": messages,
        "held_out": plan.held_out,
    }


def generate(
    config: SFTGenConfig,
    *,
    output_path: Path | None = None,
    heldout_output_path: Path | None = None,
    with_heldout: bool = False,
) -> dict[str, GenerationStats]:
    """Drive the data-generation loop. Returns per-task stats."""

    plans = resolve_task_plans(config)
    final_output = output_path or REPO_ROOT / config.output
    final_heldout = heldout_output_path

    final_output.parent.mkdir(parents=True, exist_ok=True)
    if final_heldout is not None:
        final_heldout.parent.mkdir(parents=True, exist_ok=True)

    runner = RolloutRunner()
    stats: dict[str, GenerationStats] = {plan.task_id: GenerationStats() for plan in plans}
    seen_hashes: set[str] = set()
    train_records: list[dict[str, Any]] = []
    heldout_records: list[dict[str, Any]] = []

    for plan in plans:
        for seed in plan.seeds:
            for policy_entry in plan.policies:
                for replicate in range(policy_entry.weight):
                    stats[plan.task_id].rollouts_attempted += 1
                    policy = build_policy(policy_entry.name, task_id=plan.task_id, seed=seed * 100 + replicate)
                    result = runner.run(
                        policy,
                        seed=seed,
                        task_id=plan.task_id,
                        max_steps=plan.max_steps,
                    )
                    if config.require_success and not result.success:
                        stats[plan.task_id].rollouts_failed_success += 1
                        continue
                    record = _record_for(
                        result,
                        plan=plan,
                        policy_name=policy_entry.name,
                        use_tool_calls=config.use_tool_calls,
                    )
                    if config.deduplicate:
                        digest = _hash_messages(record["messages"])
                        if digest in seen_hashes:
                            stats[plan.task_id].rollouts_dropped_duplicate += 1
                            continue
                        seen_hashes.add(digest)
                    if plan.held_out:
                        heldout_records.append(record)
                    else:
                        train_records.append(record)
                    stats[plan.task_id].rollouts_kept += 1
                    stats[plan.task_id]._reward_sum += float(record["reward"])

    if config.shuffle:
        rng = random.Random(config.shuffle_seed)
        rng.shuffle(train_records)
        rng.shuffle(heldout_records)

    LOG.info("writing %d train records to %s", len(train_records), final_output)
    with final_output.open("w") as handle:
        for record in train_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    if final_heldout is not None and (heldout_records or with_heldout):
        LOG.info("writing %d held-out records to %s", len(heldout_records), final_heldout)
        with final_heldout.open("w") as handle:
            for record in heldout_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    for task_id, task_stats in stats.items():
        kept = task_stats.rollouts_kept
        task_stats.avg_reward = round(task_stats._reward_sum / kept, 4) if kept else 0.0

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate SFT warm-start data for DroneCaptureOps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=None, help="Override output JSONL path.")
    parser.add_argument("--heldout-output", type=Path, default=None, help="Optional path for held-out trajectories.")
    parser.add_argument("--with-heldout", action="store_true", help="Also write held-out trajectories (default: skip).")
    parser.add_argument("--seeds-per-task", type=int, default=None, help="Override config's seeds_per_task.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override config's max_steps.")
    parser.add_argument("--include-tasks", default=None, help="Comma-separated task_ids to include (overrides config).")
    parser.add_argument("--exclude-tasks", default=None, help="Comma-separated task_ids to exclude.")
    parser.add_argument("--use-tool-calls", action="store_true", help="Emit assistant turns in OpenAI tool_calls dialect.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-task summary on stdout.")
    return parser.parse_args()


def _apply_cli_overrides(config: SFTGenConfig, args: argparse.Namespace) -> SFTGenConfig:
    update: dict[str, Any] = {}
    if args.seeds_per_task is not None:
        update["seeds_per_task"] = args.seeds_per_task
    if args.max_steps is not None:
        update["max_steps"] = args.max_steps
    if args.include_tasks is not None:
        update["include_tasks"] = [t.strip() for t in args.include_tasks.split(",") if t.strip()]
    if args.exclude_tasks is not None:
        update["exclude_tasks"] = [t.strip() for t in args.exclude_tasks.split(",") if t.strip()]
    if args.use_tool_calls:
        update["use_tool_calls"] = True
    if not update:
        return config
    return config.model_copy(update=update)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)

    output_path = args.output or (REPO_ROOT / config.output)
    heldout_path = args.heldout_output
    if heldout_path is None and config.held_out_tasks:
        heldout_path = output_path.parent / (output_path.stem + ".heldout.jsonl")

    started = time.time()
    stats = generate(
        config,
        output_path=output_path,
        heldout_output_path=heldout_path,
        with_heldout=args.with_heldout,
    )
    elapsed = time.time() - started

    if not args.quiet:
        kept_total = sum(s.rollouts_kept for s in stats.values())
        attempted_total = sum(s.rollouts_attempted for s in stats.values())
        print(f"\nSFT generation: {kept_total}/{attempted_total} rollouts kept in {elapsed:.1f}s")
        print(f"output: {output_path}")
        if heldout_path is not None:
            print(f"held-out output: {heldout_path}")
        print()
        print(f"{'task_id':<32} {'attempted':>10} {'kept':>6} {'failed':>7} {'dup':>5} {'avg_r':>7}")
        for task_id in sorted(stats):
            s = stats[task_id]
            held = " *" if any(plan.held_out for plan in resolve_task_plans(config) if plan.task_id == task_id) else ""
            print(
                f"{task_id + held:<32} {s.rollouts_attempted:>10} {s.rollouts_kept:>6} "
                f"{s.rollouts_failed_success:>7} {s.rollouts_dropped_duplicate:>5} {s.avg_reward:>7.3f}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
