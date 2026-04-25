"""Random baseline rollout."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def random_action(rng: random.Random) -> RawDroneAction:
    candidates = [
        RawDroneAction(tool_name="get_mission_checklist"),
        RawDroneAction(tool_name="takeoff", arguments={"altitude_m": 16}),
        RawDroneAction(tool_name="fly_to_viewpoint", arguments={"x": 8, "y": 16, "z": 16, "yaw_deg": 0, "speed_mps": 4}),
        RawDroneAction(tool_name="set_gimbal", arguments={"pitch_deg": -55, "yaw_deg": 0}),
        RawDroneAction(tool_name="capture_thermal", arguments={"label": "random thermal"}),
        RawDroneAction(tool_name="capture_rgb", arguments={"label": "random rgb"}),
        RawDroneAction(tool_name="return_home"),
        RawDroneAction(tool_name="land"),
    ]
    return rng.choice(candidates)


def main() -> None:
    rng = random.Random(3)
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=3)
    for _ in range(12):
        obs = env.step(random_action(rng))
        if obs.done:
            break
    print(json.dumps({"reward": obs.reward, "done": obs.done, "status": obs.checklist_status.model_dump()}, indent=2))


if __name__ == "__main__":
    main()
