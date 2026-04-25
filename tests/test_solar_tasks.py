from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS
from examples.run_task_suite import solve_task


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def safe_overview_capture(env: DroneCaptureOpsEnvironment):
    obs = env.step(act("takeoff", altitude_m=18))
    assert obs.error is None
    obs = env.step(act("fly_to_viewpoint", x=0, y=38, z=18, yaw_deg=0, speed_mps=5))
    assert obs.error is None
    obs = env.step(act("fly_to_viewpoint", x=30, y=24, z=18, yaw_deg=-90, speed_mps=5))
    assert obs.error is None
    obs = env.step(act("set_gimbal", pitch_deg=-60, yaw_deg=0))
    assert obs.error is None
    obs = env.step(act("capture_thermal", label="overview"))
    assert obs.error is None
    return obs


def test_solar_task_catalog_has_fifteen_named_tasks():
    assert len(SOLAR_TASKS) == 15
    assert {
        "basic_thermal_survey",
        "anomaly_confirmation",
        "low_battery_inspection",
        "bad_weather_recapture",
        "safety_constrained_route",
        "sparse_evidence_trap",
    } <= set(SOLAR_TASKS)


def test_all_solar_tasks_reset_as_openenv_compatible_observations():
    for task_id, spec in SOLAR_TASKS.items():
        env = DroneCaptureOpsEnvironment()
        obs = env.reset(seed=7, task=task_id)

        assert obs.mission.task_id == task_id
        assert obs.mission.task_name == spec.name
        assert obs.mission.required_rows == list(spec.required_rows)
        assert "submit_evidence_pack" in obs.available_tools
        assert obs.metadata["domain"] == "solar"
        assert env.state.domain == "solar"


def test_task_id_alias_matches_task_reset_argument():
    env_a = DroneCaptureOpsEnvironment()
    env_b = DroneCaptureOpsEnvironment()

    obs_a = env_a.reset(seed=7, task="anomaly_confirmation")
    obs_b = env_b.reset(seed=7, task_id="anomaly_confirmation")

    assert obs_a.mission == obs_b.mission
    assert obs_a.site_map == obs_b.site_map


def test_unknown_solar_task_is_rejected():
    env = DroneCaptureOpsEnvironment()

    try:
        env.reset(seed=7, task="not_a_real_task")
    except ValueError as exc:
        assert "unsupported solar task" in str(exc)
    else:
        raise AssertionError("unknown task should raise ValueError")


def test_task_observations_do_not_leak_hidden_defects_before_sensing():
    for task_id in SOLAR_TASKS:
        env = DroneCaptureOpsEnvironment()
        obs = env.reset(seed=7, task=task_id)
        obs_json = obs.model_dump_json()
        state_json = env.state.model_dump_json()

        for defect in env.debug_world.hidden_defects:
            assert defect.defect_id not in obs_json
            assert defect.defect_id not in state_json
            assert defect.defect_type not in obs_json
            assert defect.defect_type not in state_json


def test_scripted_rollout_completes_every_solar_task():
    for task_id in SOLAR_TASKS:
        obs = solve_task(task_id)

        assert obs.done is True
        assert obs.checklist_status.complete is True
        assert obs.action_result["accepted"] is True
        assert obs.reward > 0.75


def test_anomaly_confirmation_requires_rgb_of_same_target_row():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7, task="anomaly_confirmation")
    obs = safe_overview_capture(env)

    assert obs.checklist_status.anomalies_detected == ["hotspot_B6"]
    assert obs.checklist_status.anomaly_targets == {"hotspot_B6": "row_B6"}

    obs = env.step(act("set_zoom", zoom_level=2.0))
    assert obs.error is None
    obs = env.step(act("fly_to_viewpoint", x=10, y=24, z=18, yaw_deg=0, speed_mps=5))
    assert obs.error is None
    obs = env.step(act("capture_rgb", label="wrong row"))
    assert obs.error is None
    assert obs.checklist_status.anomaly_rgb_pairs == {}

    obs = env.step(act("fly_to_viewpoint", x=45, y=0, z=16, yaw_deg=180, speed_mps=5))
    assert obs.error is None
    obs = env.step(act("capture_rgb", label="correct row"))
    assert obs.error is None
    assert obs.checklist_status.anomaly_rgb_pairs == {"hotspot_B6": "IMG-R-003"}


def test_sparse_evidence_trap_rejects_premature_partial_report():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="sparse_evidence_trap")

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=38, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=10, y=24, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("set_gimbal", pitch_deg=-60, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="partial"))
    assert obs.last_capture.targets_visible != obs.mission.required_rows

    obs = env.step(act("submit_evidence_pack", summary="partial", photo_ids=["IMG-T-001"], findings=[]))

    assert obs.done is True
    assert obs.action_result["accepted"] is False
    assert any("missing thermal row citations" in warning for warning in obs.action_result["warnings"])


def test_privacy_task_blocks_capture_from_privacy_zone():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="privacy_zone_capture")

    env.step(act("takeoff", altitude_m=18))
    obs = env.step(act("fly_to_viewpoint", x=30, y=16, z=12, yaw_deg=-90, speed_mps=5))
    assert obs.error is None

    obs = env.step(act("capture_rgb", label="privacy violation"))

    assert "privacy_capture_violation" in obs.error
    assert obs.reward_breakdown.safety_gate == 0.0
