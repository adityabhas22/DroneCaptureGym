"""Baseline policies for rollout, tracing, and evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from dronecaptureops.core.models import DroneObservation, RawDroneAction


def act(tool_name: str, **arguments) -> RawDroneAction:
    """Build a typed environment action."""

    return RawDroneAction(tool_name=tool_name, arguments=arguments)


class ActionPolicy(Protocol):
    """Policy interface used by RolloutRunner."""

    name: str

    def next_action(self, observation: DroneObservation, history: list[dict]) -> RawDroneAction:
        """Return one action for the current observation."""


@dataclass
class RandomPolicy:
    """Random baseline over a bounded set of valid-ish actions."""

    seed: int = 0
    name: str = "random"

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def next_action(self, observation: DroneObservation, history: list[dict]) -> RawDroneAction:
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
class WeakScriptedPolicy:
    """Weak baseline that gathers partial evidence and submits too early.

    Captures only the north half of the row block, never returns home, and
    submits an incomplete pack — exactly the demo failure mode.
    """

    name: str = "weak_scripted"

    def next_action(self, observation: DroneObservation, history: list[dict]) -> RawDroneAction:
        step = len(history)
        sequence = [
            act("get_mission_checklist"),
            act("list_assets"),
            act("takeoff", altitude_m=18),
            act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5),
            act("fly_to_viewpoint", x=30, y=12, z=30, yaw_deg=-90, speed_mps=5),
            act("set_gimbal", pitch_deg=-75, yaw_deg=0),
            act("capture_thermal", label="single overview"),
            act("inspect_capture", photo_id="IMG-T-001"),
            act(
                "submit_evidence_pack",
                summary="Partial overview submitted.",
                photo_ids=["IMG-T-001"],
                findings=[],
            ),
        ]
        if step < len(sequence):
            return sequence[step]
        return act("hover", seconds=1)


@dataclass
class ScriptedPolicy:
    """Scripted baseline that completes the multi-viewpoint solar mission.

    Two thermal captures from staggered overhead viewpoints cover all five
    rows, RGB close-ups capture anomaly context if detected, and the
    evidence pack cites the thermal photos for coverage and the RGB photos
    for issues.
    """

    name: str = "scripted"

    def next_action(self, observation: DroneObservation, history: list[dict]) -> RawDroneAction:
        step = len(history)
        sequence: list[RawDroneAction] = [
            act("get_mission_checklist"),
            act("list_assets"),
            act("takeoff", altitude_m=18),
            # Bypass the substation no-fly zone via the north corridor.
            act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5),
            # North overview: covers rows B6-B8 from y=24 looking south.
            act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5),
            act("set_camera_source", source="thermal"),
            act("set_gimbal", pitch_deg=-56, yaw_deg=0),
            act("capture_thermal", label="thermal overview B6-B8"),
            act("inspect_capture", photo_id="IMG-T-001"),
            # South overview: covers rows B4-B6 from y=-24 looking north.
            act("fly_to_viewpoint", x=30, y=-24, z=22, yaw_deg=90, speed_mps=5),
            act("set_gimbal", pitch_deg=-56, yaw_deg=0),
            act("capture_thermal", label="thermal overview B4-B6"),
            act("inspect_capture", photo_id="IMG-T-002"),
            # RGB anomaly context: south close-up first (we're already south).
            act("fly_to_viewpoint", x=30, y=-24, z=14, yaw_deg=90, speed_mps=4),
            act("set_camera_source", source="rgb"),
            act("set_gimbal", pitch_deg=-45, yaw_deg=0),
            act("capture_rgb", label="rgb close-up south"),
            # RGB north close-up.
            act("fly_to_viewpoint", x=30, y=24, z=14, yaw_deg=-90, speed_mps=4),
            act("set_gimbal", pitch_deg=-45, yaw_deg=0),
            act("capture_rgb", label="rgb close-up north"),
            act("estimate_return_margin"),
            act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5),
            act("return_home"),
            act("land"),
        ]
        if step < len(sequence):
            return sequence[step]

        photo_ids = [capture.photo_id for capture in observation.capture_log]
        thermal_ids = [capture.photo_id for capture in observation.capture_log if capture.sensor == "thermal"]
        rgb_ids = [capture.photo_id for capture in observation.capture_log if capture.sensor == "rgb"]
        findings = [
            {"finding": anomaly, "photo_ids": rgb_ids or photo_ids}
            for anomaly in observation.checklist_status.anomalies_detected
        ]
        return act(
            "submit_evidence_pack",
            summary=(
                "Rows B4-B8 inspected via two overhead thermal passes and RGB "
                "close-ups; returned home with battery reserve."
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


def get_policy(name: str, *, seed: int = 0) -> ActionPolicy:
    """Return a named baseline policy."""

    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "weak_scripted":
        return WeakScriptedPolicy()
    if name == "scripted":
        return ScriptedPolicy()
    available = ", ".join(["random", "weak_scripted", "scripted"])
    raise ValueError(f"unknown policy {name!r}; available policies: {available}")
