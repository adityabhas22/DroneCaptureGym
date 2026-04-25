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
        act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5),
        act("fly_to_viewpoint", x=30, y=24, z=18, yaw_deg=-90, speed_mps=5),
        act("set_gimbal", pitch_deg=-60, yaw_deg=0),
        act("capture_thermal", label="thermal overview B4-B8"),
        act("inspect_capture", photo_id="IMG-T-001"),
        act("fly_to_viewpoint", x=30, y=16, z=12, yaw_deg=-90, speed_mps=4),
        act("capture_rgb", label="rgb anomaly context"),
        act("return_home"),
        act("land"),
    ]
    for action in actions:
        obs = env.step(action)
    photo_ids = [capture.photo_id for capture in obs.capture_log]
    findings = [{"finding": anomaly, "photo_ids": photo_ids} for anomaly in obs.checklist_status.anomalies_detected]
    obs = env.step(
        act(
            "submit_evidence_pack",
            summary="Rows B4-B8 inspected with thermal overview and RGB context for detected anomalies.",
            photo_ids=photo_ids,
            findings=findings,
        )
    )
    print(json.dumps({"reward": obs.reward, "done": obs.done, "status": obs.checklist_status.model_dump()}, indent=2))


if __name__ == "__main__":
    main()
