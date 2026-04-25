"""Run deterministic scripted rollouts for all solar mission tasks."""

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
        raise RuntimeError(f"{action.tool_name} failed: {obs.error}")
    return obs


def row_y(obs: DroneObservation, row_id: str) -> float:
    for asset in obs.visible_assets:
        if asset.asset_id == row_id:
            return asset.center_y
    raise KeyError(row_id)


def solve_task(task_id: str, seed: int = 7) -> DroneObservation:
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=seed, task=task_id)

    obs = step(env, act("get_mission_checklist"))
    obs = step(env, act("takeoff", altitude_m=18))
    obs = step(env, act("fly_to_viewpoint", x=0, y=38, z=18, yaw_deg=0, speed_mps=5))

    if task_id == "bad_weather_recapture":
        obs = step(env, act("fly_to_viewpoint", x=10, y=24, z=18, yaw_deg=0, speed_mps=5))
        obs = step(env, act("set_gimbal", pitch_deg=-60, yaw_deg=0))
        obs = step(env, act("capture_thermal", label="weather degraded first attempt"))
        obs = step(env, act("inspect_capture", photo_id=obs.last_capture.photo_id))
        obs = step(env, act("fly_to_viewpoint", x=0, y=38, z=18, yaw_deg=0, speed_mps=5))

    obs = step(env, act("fly_to_viewpoint", x=30, y=24, z=18, yaw_deg=-90, speed_mps=5))
    obs = step(env, act("set_gimbal", pitch_deg=-60, yaw_deg=0))
    obs = step(env, act("capture_thermal", label=f"{task_id} thermal overview"))
    obs = step(env, act("inspect_capture", photo_id=obs.last_capture.photo_id))

    if obs.mission.rgb_closeup_for_anomalies:
        obs = step(env, act("set_zoom", zoom_level=2.0))
        for anomaly in obs.checklist_status.anomalies_detected:
            target_id = obs.checklist_status.anomaly_targets[anomaly]
            obs = step(
                env,
                act(
                    "fly_to_viewpoint",
                    x=45,
                    y=row_y(obs, target_id),
                    z=16,
                    yaw_deg=180,
                    speed_mps=5,
                ),
            )
            obs = step(env, act("set_gimbal", pitch_deg=-60, yaw_deg=0))
            obs = step(env, act("capture_rgb", label=f"rgb confirmation {anomaly}"))
            obs = step(env, act("inspect_capture", photo_id=obs.last_capture.photo_id))

    if obs.telemetry.pose.x > 35:
        obs = step(env, act("fly_to_viewpoint", x=45, y=38, z=18, yaw_deg=180, speed_mps=5))
    obs = step(env, act("fly_to_viewpoint", x=0, y=38, z=18, yaw_deg=180, speed_mps=5))
    obs = step(env, act("return_home"))
    obs = step(env, act("land"))

    photo_ids = [capture.photo_id for capture in obs.capture_log]
    anomalies = obs.checklist_status.anomalies_detected
    if anomalies:
        summary = "Inspected rows B4-B8 and confirmed anomalies: " + ", ".join(anomalies)
    else:
        summary = "Inspected rows B4-B8 with no detected thermal anomalies."
    findings = [
        {
            "finding": anomaly,
            "target_id": obs.checklist_status.anomaly_targets.get(anomaly),
            "photo_ids": photo_ids,
        }
        for anomaly in anomalies
    ]
    return env.step(act("submit_evidence_pack", summary=summary, photo_ids=photo_ids, findings=findings))


def main() -> None:
    results = {}
    for task_id in SOLAR_TASKS:
        obs = solve_task(task_id)
        results[task_id] = {
            "reward": obs.reward,
            "done": obs.done,
            "complete": obs.checklist_status.complete,
            "battery_pct": obs.telemetry.battery.level_pct,
            "warnings": obs.action_result.get("warnings", []),
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
