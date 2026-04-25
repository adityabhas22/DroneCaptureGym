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
    """Weak baseline that gathers partial evidence and submits too early."""

    name: str = "weak_scripted"

    def next_action(self, observation: DroneObservation, history: list[dict]) -> RawDroneAction:
        step = len(history)
        sequence = [
            act("get_mission_checklist"),
            act("list_assets"),
            act("takeoff", altitude_m=18),
            act("move_to_asset", asset_id="row_B6", standoff_bucket="far", speed_mps=5),
            act("point_camera_at", asset_id="row_B6"),
            act("capture_thermal", label="single overview"),
            act("inspect_capture", photo_id="IMG-T-001"),
            act("submit_evidence_pack", summary="Partial overview submitted.", photo_ids=["IMG-T-001"], findings=[]),
        ]
        if step < len(sequence):
            return sequence[step]
        return act("hover", seconds=1)


@dataclass
class ScriptedPolicy:
    """Scripted baseline that uses the richer tool surface."""

    name: str = "scripted"

    def next_action(self, observation: DroneObservation, history: list[dict]) -> RawDroneAction:
        step = len(history)
        if step == 0:
            return act("get_mission_checklist")
        if step == 1:
            return act("list_assets")
        if step == 2:
            return act("takeoff", altitude_m=18)
        if step == 3:
            return act("move_to_asset", asset_id="row_B6", standoff_bucket="far", speed_mps=5)
        if step == 4:
            return act("point_camera_at", asset_id="row_B6")
        if step == 5:
            return act("set_camera_source", source="thermal")
        if step == 6:
            return act("capture_thermal", label="thermal overview B4-B8")
        if step == 7:
            return act("inspect_capture", photo_id="IMG-T-001")
        if step == 8:
            return act("move_to_asset", asset_id="row_B6", standoff_bucket="mid", speed_mps=4)
        if step == 9:
            return act("point_camera_at", asset_id="row_B6")
        if step == 10:
            return act("set_camera_source", source="rgb")
        if step == 11:
            return act("capture_rgb", label="rgb anomaly context")
        if step == 12:
            return act("estimate_return_margin")
        if step == 13:
            return act("return_home")
        if step == 14:
            return act("land")
        photo_ids = [capture.photo_id for capture in observation.capture_log]
        findings = [
            {"finding": anomaly, "photo_ids": photo_ids}
            for anomaly in observation.checklist_status.anomalies_detected
        ]
        return act(
            "submit_evidence_pack",
            summary="Rows B4-B8 inspected with thermal overview and RGB context for detected anomalies.",
            photo_ids=photo_ids,
            findings=findings,
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
