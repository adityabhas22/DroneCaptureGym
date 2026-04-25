"""Run a short manual episode."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def main() -> None:
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7)
    print(obs.system_message)
    action = RawDroneAction(tool_name="get_mission_checklist")
    obs = env.step(action)
    print(obs.action_result)


if __name__ == "__main__":
    main()
