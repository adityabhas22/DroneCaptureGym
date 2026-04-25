"""Shared rollout runner for inference, SFT data gen, and RL trainers.

Mirrors `evaluation.rollout.RolloutRunner` but consumes the new `Policy`
protocol that takes an `AgentContext`. The same runner can drive a
scripted solver, an OpenAI-backed policy, or an HF chat-template
local model — they're swappable at the policy boundary.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from dronecaptureops.agent.policies import AgentContext, Policy
from dronecaptureops.agent.messages import trajectory_to_messages
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.tasks.solar_tasks import SolarTaskSpec, get_solar_task


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
    parse_error: str | None = None


class RolloutResult(BaseModel):
    """Complete trajectory for one episode."""

    policy_name: str
    seed: int | None = None
    task_id: str | None = None
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
    """Run a policy through one episode and capture every transition.

    The runner is policy-agnostic: it never assumes the policy is scripted
    or model-driven. Both surface the same RawDroneAction at the
    `next_action` boundary; everything else is identical.
    """

    def __init__(self, env: DroneCaptureOpsEnvironment | None = None) -> None:
        self.env = env or DroneCaptureOpsEnvironment()

    def run(
        self,
        policy: Policy,
        *,
        seed: int | None = None,
        task_id: str | None = None,
        scenario_family: str | None = None,
        max_steps: int | None = None,
    ) -> RolloutResult:
        observation = self.env.reset(seed=seed, task=task_id, scenario_family=scenario_family)
        initial_observation = observation.model_dump(mode="json")
        limit = max_steps or int(observation.state_summary.get("remaining_steps") or 40)
        context = AgentContext()
        trajectory: list[RolloutStep] = []
        previous_breakdown = observation.reward_breakdown.model_dump(mode="json")

        for step_number in range(1, limit + 1):
            action_or_error = self._next_action(policy, observation, context)
            action: RawDroneAction | None = None
            if isinstance(action_or_error, ActionValidationError):
                next_observation = self.env.step({"tool_name": "invalid", "arguments": {}})
                parse_error = str(action_or_error)
                action_payload = {"tool_name": "invalid", "arguments": {}}
            else:
                action = action_or_error
                next_observation = self.env.step(action)
                parse_error = None
                action_payload = action.model_dump(mode="json")

            breakdown = next_observation.reward_breakdown.model_dump(mode="json")
            trajectory.append(
                RolloutStep(
                    step=step_number,
                    observation=observation.model_dump(mode="json"),
                    action=action_payload,
                    next_observation=next_observation.model_dump(mode="json"),
                    reward=float(next_observation.reward or 0.0),
                    reward_breakdown=breakdown,
                    reward_delta=_reward_delta(previous_breakdown, breakdown),
                    done=next_observation.done,
                    action_result=next_observation.action_result,
                    warnings=list(next_observation.warnings),
                    parse_error=parse_error,
                )
            )
            if action is not None:
                context.append(
                    action=action,
                    observation=next_observation,
                    action_result=next_observation.action_result,
                )
            previous_breakdown = breakdown
            observation = next_observation
            if next_observation.done:
                break

        final_observation = observation.model_dump(mode="json")
        return RolloutResult(
            policy_name=getattr(policy, "name", policy.__class__.__name__),
            seed=seed,
            task_id=task_id or final_observation.get("metadata", {}).get("task_id"),
            scenario_family=scenario_family or final_observation.get("metadata", {}).get("scenario_family"),
            episode_id=final_observation.get("metadata", {}).get("episode_id"),
            success=bool(observation.done and observation.checklist_status.complete),
            steps=len(trajectory),
            total_reward=float(observation.reward or 0.0),
            reward_breakdown=observation.reward_breakdown.model_dump(mode="json"),
            trajectory=trajectory,
            initial_observation=initial_observation,
            final_observation=final_observation,
        )

    def _next_action(
        self, policy: Policy, observation: DroneObservation, context: AgentContext
    ) -> RawDroneAction | ActionValidationError:
        """Call the policy and turn parsing errors into structured failures."""

        try:
            return policy.next_action(observation, context)
        except ActionValidationError as exc:
            return exc


def trajectory_to_chat_messages(
    result: RolloutResult,
    *,
    use_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Render a complete trajectory as a chat-format messages list.

    Used by the SFT data generator to emit JSONL examples and by the
    inference CLI to log what the model actually saw + said.
    """

    initial = DroneObservation.model_validate(result.initial_observation)
    steps_payload = []
    for step in result.trajectory:
        action = RawDroneAction(
            tool_name=step.action.get("tool_name") or step.action.get("tool", "invalid"),
            arguments=step.action.get("arguments") or step.action.get("args", {}),
        )
        next_observation = DroneObservation.model_validate(step.next_observation)
        steps_payload.append({"action": action, "next_observation": next_observation})

    runner_env = DroneCaptureOpsEnvironment()
    runner_env.reset(
        seed=result.seed,
        task=result.task_id,
        scenario_family=result.scenario_family,
    )
    task_spec: SolarTaskSpec | None = None
    if result.task_id:
        try:
            task_spec = get_solar_task(result.task_id)
        except ValueError:
            task_spec = None
    return trajectory_to_messages(
        initial_observation=initial,
        steps=steps_payload,
        registry=runner_env._tools,  # noqa: SLF001 — public-ish read of the live registry
        world=runner_env.debug_world,
        task=task_spec,
        use_tool_calls=use_tool_calls,
    )


def _reward_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key in sorted(set(before) | set(after)):
        before_value = before.get(key, 0.0)
        after_value = after.get(key, 0.0)
        if isinstance(before_value, int | float) and isinstance(after_value, int | float):
            delta = float(after_value) - float(before_value)
            if abs(delta) > 1e-9:
                deltas[key] = round(delta, 6)
    return deltas
