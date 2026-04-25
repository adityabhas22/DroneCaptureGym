"""Robustness tests: bad agent inputs do not crash the environment."""

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_string_for_numeric_arg_returns_validation_error():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step(act("takeoff", altitude_m="high"))

    assert obs.error is not None
    assert "altitude_m" in obs.error
    assert obs.reward_breakdown.format_validity == 0.0


def test_null_for_required_arg_returns_validation_error():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step(act("fly_to_viewpoint", x=10, y=10, z=None))

    assert obs.error is not None
    assert "z" in obs.error


def test_unsupported_camera_source_is_validation_error_not_crash():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step(act("set_camera_source", source="lidar"))

    assert obs.error is not None
    assert obs.error  # specific message includes 'source must be one of'
    assert "rgb" in obs.error


def test_label_must_be_string_or_absent():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    env.step(act("takeoff", altitude_m=18))

    obs = env.step(act("capture_thermal", label=123))

    assert obs.error is not None
    assert "label" in obs.error


def test_speed_below_minimum_is_safety_violation():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    env.step(act("takeoff", altitude_m=18))

    obs = env.step(act("fly_to_viewpoint", x=5, y=5, z=18, speed_mps=0))

    assert obs.error is not None
    assert "unsafe_speed" in obs.error


def test_repeated_invalid_actions_terminate_episode():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    for _ in range(3):
        env.step(act("takeoff", altitude_m="not_a_number"))

    assert env.state.done is True
