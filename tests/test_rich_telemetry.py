from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_rich_telemetry_transitions_through_flight_lifecycle():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7)

    assert obs.telemetry.autopilot.mode == "idle"
    assert obs.telemetry.autopilot.armed is False
    assert obs.telemetry.gps.fix_type == 3
    assert obs.telemetry.battery.level_pct == 100.0
    assert obs.state_summary["visible_asset_count"] == 5

    obs = env.step(act("takeoff", altitude_m=18))
    assert obs.telemetry.autopilot.mode == "guided"
    assert obs.telemetry.autopilot.armed is True
    assert obs.telemetry.autopilot.command_status == "completed"
    assert obs.telemetry.rangefinder.distance_m == 18
    assert obs.telemetry.battery.level_pct < 100.0

    obs = env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    assert obs.telemetry.velocity.ground_speed_mps > 0
    assert obs.telemetry.distance_flown_m > 0
    assert obs.telemetry.elapsed_time_s > 0

    obs = env.step(act("return_home"))
    assert obs.telemetry.autopilot.mode == "return_home"
    assert obs.checklist_status.returned_home is True

    obs = env.step(act("land"))
    assert obs.telemetry.autopilot.mode == "landed"
    assert obs.telemetry.autopilot.armed is False
    assert obs.telemetry.landed is True


def test_capture_creates_visible_artifact_with_camera_state():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=18, yaw_deg=-90, speed_mps=5))
    env.step(act("set_gimbal", pitch_deg=-60, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="overview"))

    assert len(obs.evidence_artifacts) == 1
    artifact = obs.evidence_artifacts[0]
    assert artifact.photo_id == "IMG-T-001"
    assert artifact.camera.active_source == "thermal"
    assert artifact.asset_ids == artifact.targets_visible
    assert "gsd_score" in artifact.quality_inputs
