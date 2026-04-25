"""Simple scripted baseline that completes the solar MVP mission."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments) -> RawDroneAction:
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def main() -> None:
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7)
    actions = [
        act("get_mission_checklist"),
        act("takeoff", altitude_m=18),
        # Bypass the substation no-fly zone via the north corridor.
        act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5),
        # North overview: rows B6-B8.
        act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5),
        act("set_camera_source", source="thermal"),
        act("set_gimbal", pitch_deg=-56, yaw_deg=0),
        act("capture_thermal", label="thermal overview B6-B8"),
        # South overview: rows B4-B6.
        act("fly_to_viewpoint", x=30, y=-24, z=22, yaw_deg=90, speed_mps=5),
        act("set_gimbal", pitch_deg=-56, yaw_deg=0),
        act("capture_thermal", label="thermal overview B4-B6"),
        # RGB anomaly context (south side).
        act("fly_to_viewpoint", x=30, y=-24, z=14, yaw_deg=90, speed_mps=4),
        act("set_camera_source", source="rgb"),
        act("set_gimbal", pitch_deg=-45, yaw_deg=0),
        act("capture_rgb", label="rgb close-up south"),
        # Return home + land.
        act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5),
        act("return_home"),
        act("land"),
    ]
    for action in actions:
        obs = env.step(action)
    photo_ids = [capture.photo_id for capture in obs.capture_log]
    thermal_ids = [capture.photo_id for capture in obs.capture_log if capture.sensor == "thermal"]
    findings = [
        {"finding": anomaly, "photo_ids": photo_ids}
        for anomaly in obs.checklist_status.anomalies_detected
    ]
    obs = env.step(
        act(
            "submit_evidence_pack",
            summary="Rows B4-B8 inspected via two thermal overviews + RGB anomaly context.",
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
    )
    print(json.dumps({"reward": obs.reward, "done": obs.done, "status": obs.checklist_status.model_dump()}, indent=2))


if __name__ == "__main__":
    main()
