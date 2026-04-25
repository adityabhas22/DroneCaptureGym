from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_single_overhead_capture_does_not_cover_all_rows():
    """A single thermal capture covers only part of the row block.

    Real thermal cameras have ~30° vertical FOV, so the agent must take
    multiple staggered captures to cover all five rows. This test pins the
    invariant that prevents the "one-shot wins" regression.
    """

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="overview north"))

    assert obs.error is None
    assert obs.last_capture is not None
    assert obs.last_capture.sensor == "thermal"
    assert obs.mission is not None
    covered = set(obs.checklist_status.thermal_rows_covered)
    required = set(obs.mission.required_rows)
    assert covered, "first capture should cover at least one row"
    assert covered != required, "single overhead capture must not cover all rows"
    assert obs.reward_breakdown.target_coverage < 1.0


def test_two_staggered_captures_cover_all_rows():
    """Multi-viewpoint flow: north + south overviews together cover B4-B8."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    env.step(act("capture_thermal", label="overview north"))
    env.step(act("fly_to_viewpoint", x=30, y=-24, z=22, yaw_deg=90, speed_mps=5))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="overview south"))

    assert obs.mission is not None
    assert set(obs.checklist_status.thermal_rows_covered) == set(obs.mission.required_rows)
    assert obs.reward_breakdown.target_coverage == 1.0


def test_unknown_tool_is_invalid_but_clean():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step({"tool_name": "pilot_motor_directly", "arguments": {}})

    assert obs.error
    assert obs.reward_breakdown.format_validity == 0.0
    assert env.state.done is False
