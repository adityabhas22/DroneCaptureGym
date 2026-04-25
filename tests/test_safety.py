from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_no_fly_crossing_is_blocked_before_motion():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    env.step(act("takeoff", altitude_m=18))

    obs = env.step(act("fly_to_viewpoint", x=30, y=0, z=18, yaw_deg=0, speed_mps=5))

    assert "no_fly" in obs.error
    assert env.state.telemetry.pose.x == 0
    assert env.state.safety_violations
    assert obs.reward_breakdown.safety_gate == 0.0


def test_invalid_gimbal_is_blocked():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step(act("set_gimbal", pitch_deg=-120))

    assert "invalid_gimbal_pitch" in obs.error
