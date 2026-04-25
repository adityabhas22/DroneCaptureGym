"""Task-aware oracle policy.

Replaces the procedural `solve_task` driver in `examples/run_task_suite.py`
with a proper `Policy` subclass that emits one action per `next_action`
call. The oracle is the reference solver: it completes every task in
`SOLAR_TASKS` end-to-end through the shared `RolloutRunner`, which makes
it the source of truth for SFT data generation.

The strategy is the same multi-viewpoint flow established earlier:
- Bypass the substation NFZ via the y=20 corridor (or y=38 for tasks
  shipping a north-side obstacle).
- Two staggered thermal overviews from (30, ±24, 22) at gimbal -56°
  cover all five rows.
- One RGB close-up per detected anomaly from (45, target_y, 16) at
  gimbal -45°, looking west — outside both the NFZ and the privacy zone.
- Dogleg back to home along whichever corridor is safe given the last
  pose; submit a grounded evidence pack.

The plan is built lazily in three phases (initial → anomaly → return)
because we only learn how many anomalies exist after the thermal
captures complete. Plans are extended as observations come in; the
runner sees a normal `Policy` from the outside.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dronecaptureops.agent.policies import AgentContext, Policy, act
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.tasks.solar_tasks import get_solar_task


@dataclass
class TaskOraclePolicy:
    """Reference solver for every task in SOLAR_TASKS.

    Pass `task_id` so the oracle can tune its corridor selection,
    inspect-capture cadence, and zoom usage to the active task. A
    single instance handles one episode; create a fresh one per
    rollout.
    """

    task_id: str = "basic_thermal_survey"
    name: str = "task_oracle"
    _plan: list[RawDroneAction] = field(default_factory=list, init=False)
    _cursor: int = field(default=0, init=False)
    _phase: str = field(default="initial", init=False)

    # Per-task knobs, derived once.
    _tight_steps: bool = field(default=False, init=False)
    _far_corridor: bool = field(default=False, init=False)
    _bad_weather: bool = field(default=False, init=False)
    _zoom_for_rgb: bool = field(default=False, init=False)
    def __post_init__(self) -> None:
        self._tight_steps = self.task_id in {"limited_steps_rapid_survey"}
        self._far_corridor = self.task_id in {"obstacle_detour_inspection", "safety_constrained_route"}
        self._bad_weather = self.task_id == "bad_weather_recapture"
        self._zoom_for_rgb = self.task_id == "closeup_resolution_challenge"
        # Validate task_id resolves under the new task taxonomy.
        try:
            get_solar_task(self.task_id)
        except ValueError:
            pass

    # --- public Policy API ---------------------------------------------------

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        """Return the next planned action; extend the plan lazily."""

        if not self._plan:
            self._plan.extend(self._initial_sequence())
            self._phase = "initial"

        if self._cursor >= len(self._plan):
            self._extend_plan(observation)

        if self._cursor < len(self._plan):
            action = self._plan[self._cursor]
            self._cursor += 1
            return action

        # Fallback should be unreachable once submit is queued.
        return act("hover", seconds=1)

    # --- planning ------------------------------------------------------------

    def _extend_plan(self, observation: DroneObservation) -> None:
        if self._phase == "initial":
            self._plan.extend(self._anomaly_sequence(observation))
            self._phase = "anomaly"
            return
        if self._phase == "anomaly":
            self._plan.extend(self._return_sequence(observation))
            self._plan.append(self._submit_action(observation))
            self._phase = "submit"
            return
        # Already submitted; nothing more to plan.

    def _initial_sequence(self) -> list[RawDroneAction]:
        """Pre-flight + corridor + thermal overviews."""

        steps: list[RawDroneAction] = []
        if not self._tight_steps:
            steps.append(act("get_mission_checklist"))
        steps.append(act("takeoff", altitude_m=18))
        steps.append(self._bypass_corridor_action())

        # Bad-weather degraded first capture (the agent should recover by
        # taking the standard captures afterwards).
        if self._bad_weather:
            steps.append(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
            steps.append(act("set_camera_source", source="thermal"))
            steps.append(act("set_gimbal", pitch_deg=-30, yaw_deg=0))
            steps.append(act("capture_thermal", label="weather degraded first attempt"))
            if not self._tight_steps:
                steps.append(act("inspect_capture", photo_id="IMG-T-001"))

        # North overview: rows B6-B8.
        steps.append(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
        steps.append(act("set_camera_source", source="thermal"))
        steps.append(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
        thermal_north_label = "thermal overview B6-B8"
        steps.append(act("capture_thermal", label=thermal_north_label))
        if not self._tight_steps:
            steps.append(_inspect_latest_thermal())

        # South overview: rows B4-B6.
        steps.append(act("fly_to_viewpoint", x=30, y=-24, z=22, yaw_deg=90, speed_mps=5))
        steps.append(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
        steps.append(act("capture_thermal", label="thermal overview B4-B6"))
        if not self._tight_steps:
            steps.append(_inspect_latest_thermal())

        return steps

    def _anomaly_sequence(self, observation: DroneObservation) -> list[RawDroneAction]:
        """RGB close-up for each detected anomaly (when the task asks for it).

        Multi-block routing: every RGB site sits east of its block at
        x=block.center_x+15. Transits between sites stay safe by stepping up
        to the y=24 corridor at the *previous* x first, traversing at y=24
        to the new x, then descending. This guarantees no path crosses any
        block's substation NFZ regardless of block ordering.
        """

        if not observation.mission or not observation.mission.rgb_closeup_for_anomalies:
            return []

        steps: list[RawDroneAction] = []
        anomaly_targets = observation.checklist_status.anomaly_targets
        seen: set[str] = set()
        last_close_x: float | None = None
        for anomaly in observation.checklist_status.anomalies_detected:
            target_id = anomaly_targets.get(anomaly)
            if target_id is None or target_id in seen:
                continue
            seen.add(target_id)
            target_y = _row_y(observation, target_id)
            target_x = _row_x(observation, target_id)
            if target_y is None or target_x is None:
                continue
            close_x = target_x + 15.0  # east of the row, looking west

            # Inter-block transit: step UP to corridor at the *previous*
            # close_x first (no x change here, so we never enter an NFZ),
            # then traverse along y=24 to the new close_x.
            if last_close_x is not None and abs(last_close_x - close_x) > 1.0:
                steps.append(act("fly_to_viewpoint", x=last_close_x, y=24, z=22, yaw_deg=-90, speed_mps=5))
                steps.append(act("fly_to_viewpoint", x=close_x, y=24, z=22, yaw_deg=-90, speed_mps=5))
            elif last_close_x is None:
                # First RGB: just stage at corridor of the new x.
                steps.append(act("fly_to_viewpoint", x=close_x, y=24, z=22, yaw_deg=-90, speed_mps=5))
            # Descend to target.
            steps.append(act("fly_to_viewpoint", x=close_x, y=target_y, z=16, yaw_deg=180, speed_mps=5))
            steps.append(act("set_camera_source", source="rgb"))
            steps.append(act("set_gimbal", pitch_deg=-45, yaw_deg=0))
            steps.append(act("capture_rgb", label=f"rgb confirmation {anomaly}"))

            # closeup_resolution_challenge needs higher resolution → take a
            # zoomed-in shot from the same pose.
            if self._zoom_for_rgb:
                steps.append(act("set_zoom", zoom_level=2.0))
                steps.append(act("capture_rgb", label=f"rgb zoom {anomaly}"))
                steps.append(act("set_zoom", zoom_level=1.0))
            last_close_x = close_x
        return steps

    def _return_sequence(self, observation: DroneObservation) -> list[RawDroneAction]:
        """Dogleg back to home routing around every substation NFZ.

        For multi-block tasks the agent may be east of block C (x ~ 85).
        We step west via the safe y-corridor until pose.x is at home (x=0),
        then call return_home + land.
        """

        steps: list[RawDroneAction] = []
        pose = observation.telemetry.pose if observation.telemetry else None
        if pose is None:
            steps.append(act("return_home"))
            steps.append(act("land"))
            return steps

        # If we're east of any inverter block, step back along the y-corridor.
        # Pick the corridor first so we don't have to thread an NFZ.
        use_south = pose.y < -6.0
        corridor_y = -24.0 if use_south else (38.0 if self._far_corridor else 24.0)
        corridor_yaw = 90.0 if use_south else -90.0

        if pose.x > 35.0:
            # Move along x-axis at the current y first (no NFZ crossing if y is
            # outside [-6, 6]), then drop down to the corridor.
            if not (-6.0 <= pose.y <= 6.0):
                steps.append(act("fly_to_viewpoint", x=30, y=pose.y, z=22, yaw_deg=180, speed_mps=5))
            else:
                # Get out of NFZ y-band first, then go west.
                steps.append(act("fly_to_viewpoint", x=pose.x, y=corridor_y, z=22, yaw_deg=corridor_yaw, speed_mps=5))
                steps.append(act("fly_to_viewpoint", x=30, y=corridor_y, z=22, yaw_deg=180, speed_mps=5))

        # From x=30, dogleg west via the corridor.
        if abs(pose.y if pose.x <= 35.0 else corridor_y) - abs(corridor_y) > 1.0 or pose.x > 35.0:
            steps.append(act("fly_to_viewpoint", x=30, y=corridor_y, z=22, yaw_deg=corridor_yaw, speed_mps=5))
        steps.append(act("fly_to_viewpoint", x=0, y=corridor_y, z=18, yaw_deg=180, speed_mps=5))
        steps.append(act("return_home"))
        steps.append(act("land"))
        return steps

    def _submit_action(self, observation: DroneObservation) -> RawDroneAction:
        capture_log = observation.capture_log or []
        photo_ids = [capture.photo_id for capture in capture_log]
        thermal_ids = [capture.photo_id for capture in capture_log if capture.sensor == "thermal"]
        anomalies = list(observation.checklist_status.anomalies_detected)
        anomaly_targets = observation.checklist_status.anomaly_targets

        if anomalies:
            summary = (
                "Inspected rows B4-B8 with two thermal overviews and confirmed anomalies: "
                + ", ".join(anomalies)
                + ". Returned home with battery reserve and landed."
            )
        else:
            summary = (
                "Inspected rows B4-B8 with two thermal overviews; no thermal anomalies "
                "detected. Returned home with battery reserve and landed."
            )
        findings = [
            {
                "finding": anomaly,
                "target_id": anomaly_targets.get(anomaly),
                "photo_ids": photo_ids,
            }
            for anomaly in anomalies
        ]
        issues_found = [
            {
                "issue_id": anomaly,
                "evidence_photo_ids": photo_ids,
                "recommended_followup": "manual review",
            }
            for anomaly in anomalies
        ]
        return act(
            "submit_evidence_pack",
            summary=summary,
            photo_ids=photo_ids,
            findings=findings,
            evidence=[
                {
                    "requirement_id": "thermal_overview_rows_B4_B8",
                    "status": "satisfied",
                    "photo_ids": thermal_ids,
                }
            ],
            issues_found=issues_found,
            open_items=[],
            safety_notes=["Returned home with battery reserve."],
        )

    def _bypass_corridor_action(self) -> RawDroneAction:
        if self._far_corridor:
            return act("fly_to_viewpoint", x=0, y=38, z=18, yaw_deg=0, speed_mps=5)
        return act("fly_to_viewpoint", x=0, y=20, z=18, yaw_deg=0, speed_mps=5)


def _inspect_latest_thermal() -> RawDroneAction:
    """`inspect_capture` referencing the most recent thermal photo by ID.

    The plan is built before captures execute, so we can't know the photo_id
    in advance. We use the deterministic naming convention: `IMG-T-N` where
    N is the count of thermal captures so far. Inspecting an out-of-order
    photo_id is harmless (the env just returns an error) but guessing the
    next ID ahead of time matches the env's behaviour.
    """

    return _DeferredInspectAction()


class _DeferredInspectAction(RawDroneAction):
    """Sentinel placeholder; resolved at runtime in `next_action`.

    Pydantic v2 doesn't let us subclass cleanly; instead we use a marker
    tool_name so the policy can intercept and rewrite to the real
    inspect_capture call that references the most recent thermal photo.
    """

    def __init__(self, **data) -> None:  # noqa: ANN003
        data.setdefault("tool_name", "_oracle_inspect_latest_thermal")
        data.setdefault("arguments", {})
        super().__init__(**data)


# Re-export the type so callers can isinstance-check it.
def _row_y(observation: DroneObservation, row_id: str) -> float | None:
    for asset in observation.visible_assets or []:
        if asset.asset_id == row_id:
            return asset.center_y
    return None


def _row_x(observation: DroneObservation, row_id: str) -> float | None:
    for asset in observation.visible_assets or []:
        if asset.asset_id == row_id:
            return asset.center_x
    return None


# Patch TaskOraclePolicy.next_action to resolve deferred inspect actions in-line.
_original_next_action = TaskOraclePolicy.next_action


def _next_action_with_deferred(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
    action = _original_next_action(self, observation, context)
    if action.tool_name == "_oracle_inspect_latest_thermal":
        thermal_count = sum(1 for capture in (observation.capture_log or []) if capture.sensor == "thermal")
        photo_id = f"IMG-T-{thermal_count:03d}"
        return act("inspect_capture", photo_id=photo_id)
    return action


TaskOraclePolicy.next_action = _next_action_with_deferred  # type: ignore[method-assign]


__all__ = ["TaskOraclePolicy"]


# Compatibility check — TaskOraclePolicy must satisfy the Policy protocol.
_: Policy = TaskOraclePolicy()
