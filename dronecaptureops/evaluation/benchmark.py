"""Cross-policy regression benchmarks across suites AND task IDs.

Issue #4: produce a single, reproducible report that fans out
random / weak_scripted / scripted policies over every named scenario suite
and every task ID in `dronecaptureops.tasks.solar_tasks.SOLAR_TASKS`. Output
is both JSON (machine-readable, suitable for diffing) and Markdown (for
PR review). Score bands are checked-in alongside so regressions surface
loudly without making every reward tweak fail CI.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from dronecaptureops.evaluation.policies import ActionPolicy, get_policy
from dronecaptureops.evaluation.rollout import RolloutResult, RolloutRunner
from dronecaptureops.evaluation.suite_runner import run_suite
from dronecaptureops.generation.suites import SUITES
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS

DEFAULT_POLICIES = ("random", "weak_scripted", "scripted")
DEFAULT_TASK_SEEDS = (2101, 2202, 2301)


class BenchmarkRow(BaseModel):
    """One result row in the benchmark."""

    bucket: str  # "suite" or "task"
    bucket_name: str
    policy_name: str
    episode_id: str
    success: bool
    total_reward: float
    steps: int


class PolicyBucketSummary(BaseModel):
    """Aggregate stats for one (bucket, policy) pair."""

    bucket: str
    bucket_name: str
    policy_name: str
    episodes: int
    success_rate: float
    mean_reward: float
    min_reward: float
    max_reward: float


class BenchmarkReport(BaseModel):
    """Fan-out report across suites and task IDs."""

    rows: list[BenchmarkRow] = Field(default_factory=list)
    summaries: list[PolicyBucketSummary] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["# DroneCaptureOps Regression Benchmark", ""]
        suite_summaries = [s for s in self.summaries if s.bucket == "suite"]
        task_summaries = [s for s in self.summaries if s.bucket == "task"]
        if suite_summaries:
            lines.extend(["## Scenario Suites", "", "| suite | policy | episodes | success_rate | mean_reward |", "|---|---|---:|---:|---:|"])
            for s in suite_summaries:
                lines.append(f"| {s.bucket_name} | {s.policy_name} | {s.episodes} | {s.success_rate:.3f} | {s.mean_reward:.3f} |")
        if task_summaries:
            lines.extend(["", "## Task IDs", "", "| task_id | policy | episodes | success_rate | mean_reward |", "|---|---|---:|---:|---:|"])
            for s in task_summaries:
                lines.append(f"| {s.bucket_name} | {s.policy_name} | {s.episodes} | {s.success_rate:.3f} | {s.mean_reward:.3f} |")
        return "\n".join(lines) + "\n"


def run_benchmark(
    *,
    policy_names: tuple[str, ...] = DEFAULT_POLICIES,
    suite_names: tuple[str, ...] | None = None,
    task_ids: tuple[str, ...] | None = None,
    task_seeds: tuple[int, ...] = DEFAULT_TASK_SEEDS,
) -> BenchmarkReport:
    """Run every (policy × suite) and (policy × task × seed) combination."""

    suites = suite_names if suite_names is not None else tuple(sorted(SUITES))
    tasks = task_ids if task_ids is not None else tuple(sorted(SOLAR_TASKS))
    rows: list[BenchmarkRow] = []
    summaries: list[PolicyBucketSummary] = []
    for policy_name in policy_names:
        policy = get_policy(policy_name)
        for suite_name in suites:
            suite_rows = _suite_rows(policy, suite_name)
            rows.extend(suite_rows)
            summaries.append(_summarize("suite", suite_name, policy_name, suite_rows))
        for task_id in tasks:
            task_rows = _task_rows(policy, task_id, task_seeds)
            rows.extend(task_rows)
            summaries.append(_summarize("task", task_id, policy_name, task_rows))
    return BenchmarkReport(rows=rows, summaries=summaries)


def _suite_rows(policy: ActionPolicy, suite_name: str) -> list[BenchmarkRow]:
    report = run_suite(policy, suite=suite_name)
    return [
        BenchmarkRow(
            bucket="suite",
            bucket_name=suite_name,
            policy_name=policy.name,
            episode_id=row.episode_id,
            success=row.success,
            total_reward=row.total_reward,
            steps=row.steps,
        )
        for row in report.rows
    ]


def _task_rows(policy: ActionPolicy, task_id: str, seeds: tuple[int, ...]) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    runner = RolloutRunner()
    for seed in seeds:
        result: RolloutResult = runner.run(policy, seed=seed, task_id=task_id)
        rows.append(
            BenchmarkRow(
                bucket="task",
                bucket_name=task_id,
                policy_name=policy.name,
                episode_id=f"{task_id}:{seed}",
                success=result.success,
                total_reward=result.total_reward,
                steps=result.steps,
            )
        )
    return rows


def _summarize(bucket: str, bucket_name: str, policy_name: str, rows: list[BenchmarkRow]) -> PolicyBucketSummary:
    rewards = [row.total_reward for row in rows]
    return PolicyBucketSummary(
        bucket=bucket,
        bucket_name=bucket_name,
        policy_name=policy_name,
        episodes=len(rows),
        success_rate=mean(1.0 if row.success else 0.0 for row in rows) if rows else 0.0,
        mean_reward=mean(rewards) if rewards else 0.0,
        min_reward=min(rewards) if rewards else 0.0,
        max_reward=max(rewards) if rewards else 0.0,
    )


def check_against_bands(report: BenchmarkReport, bands: dict[str, dict[str, Any]]) -> list[str]:
    """Compare a fresh report's summaries against expected score bands.

    `bands` is keyed by f"{bucket}:{bucket_name}:{policy_name}" and each value
    is a dict with optional `mean_reward_min`, `mean_reward_max`,
    `success_rate_min`, `success_rate_max`. Returns a list of human-readable
    deviations; empty list means in-band.
    """

    deviations: list[str] = []
    for summary in report.summaries:
        key = f"{summary.bucket}:{summary.bucket_name}:{summary.policy_name}"
        band = bands.get(key)
        if band is None:
            continue
        if "mean_reward_min" in band and summary.mean_reward < band["mean_reward_min"]:
            deviations.append(f"{key}: mean_reward {summary.mean_reward:.3f} < {band['mean_reward_min']}")
        if "mean_reward_max" in band and summary.mean_reward > band["mean_reward_max"]:
            deviations.append(f"{key}: mean_reward {summary.mean_reward:.3f} > {band['mean_reward_max']}")
        if "success_rate_min" in band and summary.success_rate < band["success_rate_min"]:
            deviations.append(f"{key}: success_rate {summary.success_rate:.3f} < {band['success_rate_min']}")
        if "success_rate_max" in band and summary.success_rate > band["success_rate_max"]:
            deviations.append(f"{key}: success_rate {summary.success_rate:.3f} > {band['success_rate_max']}")
    return deviations
