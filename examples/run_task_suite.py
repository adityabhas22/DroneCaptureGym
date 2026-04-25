"""Run deterministic scripted rollouts for all solar mission tasks.

The solver uses the multi-viewpoint thermal flow required by the per-target
3D FOV camera physics: north overview from (30, 24, 22) covers rows B6-B8,
south overview from (30, -24, 22) covers rows B4-B6, and RGB anomaly
context comes from the EAST close viewpoints at x=45 (which sit outside
the substation no-fly zone and outside the privacy zone for
`privacy_zone_capture`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


def act(tool_name: str, **arguments) -> RawDroneAction:
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def step(env: DroneCaptureOpsEnvironment, action: RawDroneAction) -> DroneObservation:
    obs = env.step(action)
    if obs.error:
        raise RuntimeError(f"{action.tool_name} {action.arguments} failed: {obs.error}")
    return obs


def row_y(obs: DroneObservation, row_id: str) -> float:
    for asset in obs.visible_assets:
        if asset.asset_id == row_id:
            return asset.center_y
    raise KeyError(row_id)


def _bypass_corridor(env: DroneCaptureOpsEnvironment, task_id: str) -> None:
    """Fly to the safe northern corridor (y=38 if a task ships extra obstacles, else y=20)."""

    if task_id in {"obstacle_detour_inspection", "safety_constrained_route"}:
        step(env, act("fly_to_viewpoint", x=0, y=38, z=18, yaw_deg=0, speed_mps=5))
    else:
        step(env, act("fly_to_viewpoint", x=0, y=20, z=18, yaw_deg=0, speed_mps=5))


def _capture_thermal_overviews(env: DroneCaptureOpsEnvironment, *, inspect: bool = True) -> DroneObservation:
    """Two staggered thermal overviews covering rows B4-B8."""

    step(env, act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    step(env, act("set_camera_source", source="thermal"))
    step(env, act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    obs = step(env, act("capture_thermal", label="thermal overview B6-B8"))
    if inspect:
        step(env, act("inspect_capture", photo_id=obs.last_capture.photo_id))

    step(env, act("fly_to_viewpoint", x=30, y=-24, z=22, yaw_deg=90, speed_mps=5))
    step(env, act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    obs = step(env, act("capture_thermal", label="thermal overview B4-B6"))
    if inspect:
        step(env, act("inspect_capture", photo_id=obs.last_capture.photo_id))
    return obs


def _capture_rgb_for_anomaly(env: DroneCaptureOpsEnvironment, target_id: str, target_y: float, label: str) -> DroneObservation:
    """RGB close-up from east of the row (x=45) — outside both the NFZ and the privacy zone."""

    step(env, act("fly_to_viewpoint", x=45, y=target_y, z=16, yaw_deg=180, speed_mps=5))
    step(env, act("set_camera_source", source="rgb"))
    step(env, act("set_gimbal", pitch_deg=-45, yaw_deg=0))
    obs = step(env, act("capture_rgb", label=label))
    step(env, act("inspect_capture", photo_id=obs.last_capture.photo_id))
    return obs


def solve_task(task_id: str, seed: int = 7) -> DroneObservation:
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=seed, task=task_id)

    # Tasks with a tight step budget skip the informational warm-up calls.
    tight_step_budget = task_id in {"limited_steps_rapid_survey"}
    if not tight_step_budget:
        obs = step(env, act("get_mission_checklist"))
    obs = step(env, act("takeoff", altitude_m=18))
    _bypass_corridor(env, task_id)

    # Bad-weather task: take a degraded first capture, then recapture cleanly.
    if task_id == "bad_weather_recapture":
        step(env, act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
        step(env, act("set_camera_source", source="thermal"))
        step(env, act("set_gimbal", pitch_deg=-30, yaw_deg=0))  # bad pitch first
        first = step(env, act("capture_thermal", label="weather degraded first attempt"))
        step(env, act("inspect_capture", photo_id=first.last_capture.photo_id))

    obs = _capture_thermal_overviews(env, inspect=not tight_step_budget)

    # Anomaly RGB confirmation (skip when the task says rgb_closeup_for_anomalies=False).
    if obs.mission.rgb_closeup_for_anomalies:
        for anomaly in obs.checklist_status.anomalies_detected:
            target_id = obs.checklist_status.anomaly_targets.get(anomaly)
            if target_id is None:
                continue
            obs = _capture_rgb_for_anomaly(env, target_id, row_y(obs, target_id), f"rgb confirmation {anomaly}")
            # closeup_resolution_challenge needs higher resolution → take a closer zoom shot.
            if task_id == "closeup_resolution_challenge":
                step(env, act("set_zoom", zoom_level=2.0))
                obs = _capture_rgb_for_anomaly(
                    env,
                    target_id,
                    row_y(obs, target_id),
                    f"rgb closeup-zoom {anomaly}",
                )
                step(env, act("set_zoom", zoom_level=1.0))

    # Return to home, routing around the substation NFZ. From an RGB close-up
    # at (45, target_y, 16) we step back to x=30, then dogleg via (30, ±24, 22)
    # so the segment stays outside the NFZ y-range [-6, 6]. From a south
    # overview we use the south corridor; otherwise the north corridor.
    pose = obs.telemetry.pose
    if pose.x > 35:
        rgb_obs = step(env, act("fly_to_viewpoint", x=30, y=pose.y, z=22, yaw_deg=180, speed_mps=5))
        pose = rgb_obs.telemetry.pose
    use_south_corridor = pose.y < -6.0
    # Tasks that ship a north-side obstacle (crane corridor) need to detour
    # over the top (y=38) rather than the standard y=24 line.
    far_north_required = task_id in {"obstacle_detour_inspection"}
    if use_south_corridor:
        if abs(pose.y - (-24.0)) > 1.0:
            step(env, act("fly_to_viewpoint", x=30, y=-24, z=22, yaw_deg=90, speed_mps=5))
        step(env, act("fly_to_viewpoint", x=0, y=-24, z=18, yaw_deg=180, speed_mps=5))
    elif far_north_required:
        step(env, act("fly_to_viewpoint", x=30, y=38, z=22, yaw_deg=-90, speed_mps=5))
        step(env, act("fly_to_viewpoint", x=0, y=38, z=18, yaw_deg=180, speed_mps=5))
    else:
        if abs(pose.y - 24.0) > 1.0:
            step(env, act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
        step(env, act("fly_to_viewpoint", x=0, y=24, z=18, yaw_deg=180, speed_mps=5))
    step(env, act("return_home"))
    obs = step(env, act("land"))

    photo_ids = [capture.photo_id for capture in obs.capture_log]
    thermal_ids = [capture.photo_id for capture in obs.capture_log if capture.sensor == "thermal"]
    anomalies = obs.checklist_status.anomalies_detected
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
            "target_id": obs.checklist_status.anomaly_targets.get(anomaly),
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
    return env.step(
        act(
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
    )


def main() -> None:
    results = {}
    for task_id in SOLAR_TASKS:
        try:
            obs = solve_task(task_id)
            results[task_id] = {
                "reward": obs.reward,
                "done": obs.done,
                "complete": obs.checklist_status.complete,
                "battery_pct": obs.telemetry.battery.level_pct,
                "warnings": obs.action_result.get("warnings", []),
            }
        except RuntimeError as exc:
            results[task_id] = {"error": str(exc)}
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
