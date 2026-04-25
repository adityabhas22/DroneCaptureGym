"""Camera/physics tests: per-target FOV and family-specific defect detection."""

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import HiddenDefect, RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def _set_single_defect(env: DroneCaptureOpsEnvironment, defect: HiddenDefect) -> None:
    """Replace hidden defects on the live world for deterministic assertions."""

    env.debug_world.hidden_defects = [defect]


def test_capture_records_per_target_quality_dict():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="north"))

    assert obs.last_capture is not None
    assert obs.last_capture.per_target_quality, "capture should report per-target quality"
    assert all(0.0 <= q <= 1.0 for q in obs.last_capture.per_target_quality.values())


def test_glare_artifact_only_appears_at_low_glare_score():
    """false_positive_glare scenario: artifact surfaces in shallow-pitch shots."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2203, scenario_family="false_positive_glare")
    glare_defect_id = env.debug_world.hidden_defects[0].defect_id

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))

    # Steep gimbal: high glare_score → no false detection.
    env.step(act("set_gimbal", pitch_deg=-75, yaw_deg=0))
    steep = env.step(act("capture_thermal", label="steep"))
    assert glare_defect_id not in (steep.last_capture.detected_anomalies if steep.last_capture else [])

    # Shallow gimbal: low glare_score → false detection appears.
    env.step(act("set_gimbal", pitch_deg=-25, yaw_deg=0))
    shallow = env.step(act("capture_thermal", label="shallow"))
    assert shallow.last_capture is not None
    if shallow.last_capture.glare_score < 0.78:
        assert glare_defect_id in shallow.last_capture.detected_anomalies, (
            f"glare artifact should surface when glare_score={shallow.last_capture.glare_score} < 0.78"
        )


def test_bypass_diode_fault_requires_close_standoff():
    """A diode fault is invisible from a far overhead pass and visible from close."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2103, scenario_family="bypass_diode_fault")
    defect = env.debug_world.hidden_defects[0]

    # Far overview from north position — distance to row > 32m → no detection.
    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    far = env.step(act("capture_thermal", label="far overview"))
    assert defect.defect_id not in (far.last_capture.detected_anomalies if far.last_capture else [])

    # Close standoff to the defect target row — should detect.
    asset_y = next(a.center_y for a in env.debug_world.assets if a.asset_id == defect.target_id)
    env.step(act("fly_to_viewpoint", x=30, y=asset_y + 12.0, z=14, yaw_deg=-90, speed_mps=4))
    env.step(act("point_camera_at", asset_id=defect.target_id))
    close = env.step(act("capture_thermal", label="close detail"))
    assert close.last_capture is not None
    # Distance check: per_target_metrics confirms close standoff.
    metrics = close.last_capture.per_target_metrics.get(defect.target_id, {})
    assert metrics.get("distance_m", 999) <= 32.0
    assert defect.defect_id in close.last_capture.detected_anomalies


def test_mark_target_inspected_accepts_valid_cited_thermal():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    capture_obs = env.step(act("capture_thermal", label="north"))
    assert capture_obs.last_capture is not None
    photo_id = capture_obs.last_capture.photo_id

    # Pick a target visible in the capture.
    target_id = next(iter(capture_obs.last_capture.targets_visible))
    obs = env.step(act("mark_target_inspected", target_id=target_id, photo_ids=[photo_id]))

    assert obs.action_result["accepted"] is True
    assert target_id in env.debug_world.checklist_status.targets_acknowledged


def test_mark_target_inspected_rejects_fake_photo():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    obs = env.step(act("mark_target_inspected", target_id="row_B6", photo_ids=["IMG-T-999"]))

    assert obs.action_result["accepted"] is False
    assert any("unknown photo_ids" in str(w) for w in obs.action_result.get("warnings", []))


def test_mark_target_inspected_rejects_unknown_target():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    obs = env.step(act("mark_target_inspected", target_id="row_Z99"))

    assert obs.action_result["accepted"] is False
    assert any("unknown target_id" in str(w) for w in obs.action_result.get("warnings", []))
