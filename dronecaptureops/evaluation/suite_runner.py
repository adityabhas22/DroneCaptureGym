"""Run named scenario suites and aggregate rollout results."""

from __future__ import annotations

from statistics import mean
from typing import Any

from pydantic import BaseModel, Field

from dronecaptureops.evaluation.policies import ActionPolicy
from dronecaptureops.evaluation.rollout import RolloutResult, RolloutRunner
from dronecaptureops.generation.suites import ScenarioSuite, get_suite


class SuiteRunRow(BaseModel):
    """One suite result row."""

    suite: str
    episode_id: str
    policy_name: str
    scenario_family: str
    seed: int
    split: str
    tags: list[str] = Field(default_factory=list)
    success: bool
    total_reward: float
    reward_breakdown: dict[str, Any]
    reward_delta_totals: dict[str, float]
    steps: int
    done: bool
    artifacts: int
    safety_violations: list[str] = Field(default_factory=list)
    rollout: dict[str, Any] | None = None


class SuiteRunReport(BaseModel):
    """Aggregated report over a scenario suite."""

    suite: str
    purpose: str
    policy_name: str
    heldout: bool = False
    rows: list[SuiteRunRow]
    reward_breakdown_mean: dict[str, float]

    @property
    def episodes(self) -> int:
        return len(self.rows)

    @property
    def success_rate(self) -> float:
        return mean(1.0 if row.success else 0.0 for row in self.rows) if self.rows else 0.0

    @property
    def mean_reward(self) -> float:
        return mean(row.total_reward for row in self.rows) if self.rows else 0.0

    def to_markdown(self) -> str:
        lines = [
            f"# DroneCaptureOps Suite: {self.suite}",
            "",
            f"- policy: `{self.policy_name}`",
            f"- purpose: {self.purpose}",
            f"- heldout: `{self.heldout}`",
            f"- episodes: `{self.episodes}`",
            f"- mean_reward: `{self.mean_reward:.3f}`",
            f"- success_rate: `{self.success_rate:.3f}`",
            "",
            "## Reward Columns",
            "",
            "| component | mean |",
            "|---|---:|",
        ]
        for component, value in sorted(self.reward_breakdown_mean.items()):
            lines.append(f"| {component} | {value:.3f} |")
        lines.extend(
            [
                "",
                "## Episodes",
                "",
                "| episode | family | seed | success | reward | steps | artifacts | safety |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in self.rows:
            lines.append(
                f"| {row.episode_id} | {row.scenario_family} | {row.seed} | "
                f"{int(row.success)} | {row.total_reward:.3f} | {row.steps} | "
                f"{row.artifacts} | {len(row.safety_violations)} |"
            )
        return "\n".join(lines) + "\n"


def run_suite(
    policy: ActionPolicy,
    *,
    suite: str | ScenarioSuite,
    include_rollouts: bool = False,
) -> SuiteRunReport:
    """Run a policy over a named suite."""

    resolved = get_suite(suite) if isinstance(suite, str) else suite
    rows: list[SuiteRunRow] = []
    for episode in resolved.episodes:
        rollout = RolloutRunner().run(
            policy,
            seed=episode.seed,
            scenario_family=episode.scenario_family,
            max_steps=episode.max_steps,
        )
        rows.append(_row_from_rollout(resolved.name, episode, rollout, include_rollouts))
    return SuiteRunReport(
        suite=resolved.name,
        purpose=resolved.purpose,
        policy_name=policy.name,
        heldout=resolved.heldout,
        rows=rows,
        reward_breakdown_mean=_mean_reward_breakdown(rows),
    )


def _row_from_rollout(suite_name: str, episode, rollout: RolloutResult, include_rollout: bool) -> SuiteRunRow:
    final = rollout.final_observation
    return SuiteRunRow(
        suite=suite_name,
        episode_id=episode.episode_id,
        policy_name=rollout.policy_name,
        scenario_family=episode.scenario_family,
        seed=episode.seed,
        split=episode.split,
        tags=list(episode.tags),
        success=rollout.success,
        total_reward=rollout.total_reward,
        reward_breakdown=rollout.reward_breakdown,
        reward_delta_totals=_reward_delta_totals(rollout),
        steps=rollout.steps,
        done=bool(final.get("done")),
        artifacts=len(final.get("evidence_artifacts") or []),
        safety_violations=[
            warning for warning in final.get("warnings", [])
            if "violation" in str(warning) or "unsafe" in str(warning)
        ],
        rollout=rollout.model_dump(mode="json") if include_rollout else None,
    )


def _mean_reward_breakdown(rows: list[SuiteRunRow]) -> dict[str, float]:
    keys = sorted({key for row in rows for key, value in row.reward_breakdown.items() if isinstance(value, int | float)})
    return {
        key: mean(float(row.reward_breakdown[key]) for row in rows if isinstance(row.reward_breakdown.get(key), int | float))
        for key in keys
    }


def _reward_delta_totals(rollout: RolloutResult) -> dict[str, float]:
    totals: dict[str, float] = {}
    for step in rollout.trajectory:
        for key, value in step.reward_delta.items():
            totals[key] = round(totals.get(key, 0.0) + float(value), 6)
    return totals
