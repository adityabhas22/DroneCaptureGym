"""Shared rollout runner for evaluation, training, and trace debugging."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.evaluation.policies import ActionPolicy


class RolloutStep(BaseModel):
    """One environment transition."""

    step: int
    observation: dict[str, Any]
    action: dict[str, Any]
    next_observation: dict[str, Any]
    reward: float
    reward_breakdown: dict[str, Any]
    reward_delta: dict[str, float] = Field(default_factory=dict)
    done: bool
    action_result: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class RolloutResult(BaseModel):
    """Complete trajectory for one episode."""

    policy_name: str
    seed: int | None = None
    scenario_family: str | None = None
    episode_id: str | None = None
    success: bool
    steps: int
    total_reward: float
    reward_breakdown: dict[str, Any]
    trajectory: list[RolloutStep]
    initial_observation: dict[str, Any]
    final_observation: dict[str, Any]


class RolloutRunner:
    """Shared execution primitive for baselines, training, suites, and traces."""

    def __init__(self, env: DroneCaptureOpsEnvironment | None = None) -> None:
        self.env = env or DroneCaptureOpsEnvironment()

    def run(
        self,
        policy: ActionPolicy,
        *,
        seed: int | None = None,
        scenario_family: str | None = None,
        max_steps: int | None = None,
        task_id: str | None = None,
    ) -> RolloutResult:
        """Run a policy through one episode and capture every transition.

        When `task_id` is provided, the env is reset with task conditioning;
        otherwise the legacy seed/scenario_family path is used.
        """

        observation = self.env.reset(
            seed=seed,
            scenario_family=scenario_family,
            task_id=task_id,
        )
        initial = observation.model_dump(mode="json")
        limit = max_steps or int(observation.state_summary.get("remaining_steps") or 40)
        trajectory: list[RolloutStep] = []
        previous_breakdown = observation.reward_breakdown.model_dump(mode="json")

        for step_number in range(1, limit + 1):
            action = policy.next_action(observation, [step.model_dump(mode="json") for step in trajectory])
            next_observation = self.env.step(action)
            breakdown = next_observation.reward_breakdown.model_dump(mode="json")
            trajectory.append(
                RolloutStep(
                    step=step_number,
                    observation=observation.model_dump(mode="json"),
                    action=_action_json(action),
                    next_observation=next_observation.model_dump(mode="json"),
                    reward=float(next_observation.reward or 0.0),
                    reward_breakdown=breakdown,
                    reward_delta=_reward_delta(previous_breakdown, breakdown),
                    done=next_observation.done,
                    action_result=next_observation.action_result,
                    warnings=next_observation.warnings,
                )
            )
            previous_breakdown = breakdown
            observation = next_observation
            if next_observation.done:
                break

        final = observation.model_dump(mode="json")
        final_breakdown = observation.reward_breakdown.model_dump(mode="json")
        return RolloutResult(
            policy_name=policy.name,
            seed=seed,
            scenario_family=scenario_family or observation.metadata.get("scenario_family"),
            episode_id=observation.metadata.get("episode_id"),
            success=bool(observation.done and observation.checklist_status.complete),
            steps=len(trajectory),
            total_reward=float(observation.reward or 0.0),
            reward_breakdown=final_breakdown,
            trajectory=trajectory,
            initial_observation=initial,
            final_observation=final,
        )


def _action_json(action: RawDroneAction) -> dict[str, Any]:
    return action.model_dump(mode="json")


def _reward_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    """Compute numeric reward deltas for all current and future components."""

    deltas: dict[str, float] = {}
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key, 0.0)
        after_value = after.get(key, 0.0)
        if isinstance(before_value, int | float) and isinstance(after_value, int | float):
            delta = float(after_value) - float(before_value)
            if abs(delta) > 1e-9:
                deltas[key] = round(delta, 6)
    return deltas
