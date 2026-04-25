"""Policy adapters used by the shared rollout harness.

Every policy implements the same `next_action(observation, context) -> RawDroneAction`
contract. Inference, SFT data generation, and RL trainers all run rollouts
through the same Policy/RolloutRunner pair, so a model wired through
`OpenAIChatPolicy` and a `ScriptedPolicy` are interchangeable from the
runner's perspective.

This module defines:
- `Policy` (Protocol)
- `AgentContext` (history view passed to policies each step)
- `ScriptedPolicy` and `RandomPolicy` reimplementations that share
  state with the agent runner (the original copies live in
  `evaluation/policies.py` for backward compat).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from dronecaptureops.core.models import DroneObservation, RawDroneAction


@dataclass
class AgentContext:
    """Per-rollout history visible to a policy.

    Kept lightweight: each step records the action taken, the resulting
    observation summary, and the action_result. Policies that want richer
    history can re-render via `dronecaptureops.agent.observation`.
    """

    history: list[dict[str, Any]] = field(default_factory=list)

    def append(self, *, action: RawDroneAction, observation: DroneObservation, action_result: dict[str, Any]) -> None:
        self.history.append(
            {
                "action": {"tool_name": action.tool_name, "arguments": dict(action.arguments)},
                "step": len(self.history) + 1,
                "reward": float(observation.reward or 0.0),
                "done": observation.done,
                "action_result": dict(action_result),
                "warnings": list(observation.warnings),
            }
        )


class Policy(Protocol):
    """Common interface every policy must satisfy.

    `name` is used by the rollout harness for logging and trace artifacts.
    `next_action` returns the next tool call given the current observation
    and the policy's view of history. Implementations may inspect
    `context.history` for prior actions and rewards.
    """

    name: str

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction: ...


def act(tool_name: str, **arguments: Any) -> RawDroneAction:
    """Convenience constructor for tool-call actions."""

    return RawDroneAction(tool_name=tool_name, arguments=arguments)


@dataclass
class RandomPolicy:
    """Random baseline over a bounded set of valid-ish actions."""

    seed: int = 0
    name: str = "random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        assets = observation.visible_assets or []
        asset_id = self._rng.choice(assets).asset_id if assets else "row_B6"
        captures = observation.capture_log
        choices = [
            act("get_mission_checklist"),
            act("list_assets"),
            act("takeoff", altitude_m=self._rng.choice([12, 16, 18, 22])),
            act("move_to_asset", asset_id=asset_id, standoff_bucket=self._rng.choice(["far", "mid", "close"]), speed_mps=4),
            act("point_camera_at", asset_id=asset_id),
            act("set_camera_source", source=self._rng.choice(["rgb", "thermal"])),
            act("capture_thermal", label="random thermal"),
            act("capture_rgb", label="random rgb"),
            act("estimate_return_margin"),
            act("return_home"),
            act("land"),
        ]
        if captures:
            choices.append(act("inspect_capture", photo_id=captures[-1].photo_id))
        return self._rng.choice(choices)


@dataclass
class ScriptedPolicy:
    """Multi-viewpoint scripted solver that completes the canonical solar mission.

    Mirrors `evaluation.policies.ScriptedPolicy` but accepts the new
    `AgentContext` signature so it slots into the agent harness without an
    adapter shim.
    """

    name: str = "scripted"

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        step = len(context.history)
        sequence: list[RawDroneAction] = [
            act("get_mission_checklist"),
            act("list_assets"),
            act("takeoff", altitude_m=18),
            act("fly_to_viewpoint", x=0, y=20, z=18, yaw_deg=0, speed_mps=5),
            act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5),
            act("set_camera_source", source="thermal"),
            act("set_gimbal", pitch_deg=-56, yaw_deg=0),
            act("capture_thermal", label="thermal overview B6-B8"),
            act("inspect_capture", photo_id="IMG-T-001"),
            act("fly_to_viewpoint", x=30, y=-24, z=22, yaw_deg=90, speed_mps=5),
            act("set_gimbal", pitch_deg=-56, yaw_deg=0),
            act("capture_thermal", label="thermal overview B4-B6"),
            act("inspect_capture", photo_id="IMG-T-002"),
            act("fly_to_viewpoint", x=30, y=-24, z=14, yaw_deg=90, speed_mps=4),
            act("set_camera_source", source="rgb"),
            act("set_gimbal", pitch_deg=-45, yaw_deg=0),
            act("capture_rgb", label="rgb close-up south"),
            act("fly_to_viewpoint", x=30, y=24, z=14, yaw_deg=-90, speed_mps=4),
            act("set_gimbal", pitch_deg=-45, yaw_deg=0),
            act("capture_rgb", label="rgb close-up north"),
            act("fly_to_viewpoint", x=0, y=24, z=18, yaw_deg=180, speed_mps=5),
            act("return_home"),
            act("land"),
        ]
        if step < len(sequence):
            return sequence[step]

        photo_ids = [capture.photo_id for capture in observation.capture_log]
        thermal_ids = [c.photo_id for c in observation.capture_log if c.sensor == "thermal"]
        rgb_ids = [c.photo_id for c in observation.capture_log if c.sensor == "rgb"]
        findings = [
            {"finding": anomaly, "target_id": observation.checklist_status.anomaly_targets.get(anomaly), "photo_ids": rgb_ids or photo_ids}
            for anomaly in observation.checklist_status.anomalies_detected
        ]
        return act(
            "submit_evidence_pack",
            summary=(
                "Rows B4-B8 inspected via two thermal overviews and RGB close-ups; "
                "returned home with battery reserve."
            ),
            photo_ids=photo_ids,
            findings=findings,
            evidence=[
                {
                    "requirement_id": "thermal_overview_rows_B4_B8",
                    "status": "satisfied",
                    "photo_ids": thermal_ids,
                }
            ],
            safety_notes=["Returned home with battery reserve."],
        )
