from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_step_executes_high_level_capture_flow():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=18, yaw_deg=-90, speed_mps=5))
    env.step(act("set_gimbal", pitch_deg=-60, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="overview"))

    assert obs.error is None
    assert obs.last_capture is not None
    assert obs.last_capture.sensor == "thermal"
    assert set(obs.checklist_status.thermal_rows_covered) == set(obs.mission.required_rows)
    assert obs.reward_breakdown.target_coverage == 1.0


def test_unknown_tool_is_invalid_but_clean():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step({"tool_name": "pilot_motor_directly", "arguments": {}})

    assert obs.error
    assert obs.reward_breakdown.format_validity == 0.0
    assert env.state.done is False
