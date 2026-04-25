"""Tests for the corrected task-conditioned solar benchmark.

The benchmark intentionally avoids versioned task buckets. Every task that
stays in the catalog should be constructible through the OpenEnv harness and
should have at least one simulator/reward mechanic behind it.
"""

from __future__ import annotations

import pytest

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction
from dronecaptureops.evaluation.policies import RandomPolicy, ScriptedPolicy
from dronecaptureops.evaluation.suite_runner import run_suite
from dronecaptureops.generation.scenario_generator import ScenarioGenerator
from dronecaptureops.generation.suites import get_suite
from dronecaptureops.rewards.report_grounding import validate_evidence_report
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


REMOVED_LOW_VALUE_TASK_IDS: tuple[str, ...] = (
    "bad_weather_recapture",
    "safety_constrained_route",
    "sparse_evidence_trap",
    "closeup_resolution_challenge",
    "edge_row_focus",
    "return_home_compliance",
    "limited_steps_rapid_survey",
    "report_grounding_audit",
    "string_outage_survey",
    "cracked_glass_closeup",
    "low_contrast_recapture",
    "boundary_aware_closeup",
    "adaptive_battery_reserve",
)


def act(tool_name: str, **arguments) -> RawDroneAction:
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def start_from_north_corridor(env: DroneCaptureOpsEnvironment):
    obs = env.step(act("takeoff", altitude_m=18))
    assert obs.error is None
    obs = env.step(act("fly_to_viewpoint", x=0, y=20, z=18, yaw_deg=0, speed_mps=5))
    assert obs.error is None
    return obs


def test_solar_task_catalog_replaces_low_value_reskins():
    assert len(SOLAR_TASKS) == 30
    assert not (set(REMOVED_LOW_VALUE_TASK_IDS) & set(SOLAR_TASKS))


@pytest.mark.parametrize("task_id", tuple(SOLAR_TASKS))
def test_task_resets_into_a_well_formed_observation(task_id: str):
    spec = SOLAR_TASKS[task_id]
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7, task=task_id)

    assert obs.mission.task_id == task_id
    assert obs.mission.task_name == spec.name
    assert obs.mission.required_rows == list(spec.required_rows)
    assert obs.mission.min_capture_quality == spec.min_capture_quality
    assert obs.mission.min_rgb_quality == spec.min_rgb_quality
    assert obs.mission.min_report_grounding_score == spec.min_report_grounding_score
    assert "submit_evidence_pack" in obs.available_tools
    assert env.state.domain == "solar"
    assert env.debug_world.max_steps == spec.max_steps


@pytest.mark.parametrize("task_id", tuple(SOLAR_TASKS))
def test_task_battery_weather_and_zones_match_the_spec(task_id: str):
    spec = SOLAR_TASKS[task_id]
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7, task=task_id)

    assert obs.telemetry.battery.level_pct == pytest.approx(spec.initial_battery_pct)
    if spec.weather_wind_mps is not None:
        assert env.debug_world.weather.wind_mps == pytest.approx(spec.weather_wind_mps)
    if spec.weather_visibility is not None:
        assert env.debug_world.weather.visibility == pytest.approx(spec.weather_visibility)
    extra_zone_ids = {zone.zone_id for zone in spec.extra_zones}
    if extra_zone_ids:
        actual_zone_ids = {zone.zone_id for zone in env.debug_world.airspace_zones}
        assert extra_zone_ids <= actual_zone_ids


@pytest.mark.parametrize("task_id", tuple(SOLAR_TASKS))
def test_task_observation_does_not_leak_hidden_defects(task_id: str):
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7, task=task_id)
    obs_json = obs.model_dump_json()
    state_json = env.state.model_dump_json()

    for defect in env.debug_world.hidden_defects:
        assert defect.defect_id not in obs_json, f"defect_id {defect.defect_id} leaked in obs"
        assert defect.defect_id not in state_json, f"defect_id {defect.defect_id} leaked in visible state"


def test_solar_tasks_suite_has_one_episode_per_catalog_task():
    suite = get_suite("solar_tasks")

    assert len(suite.episodes) == len(SOLAR_TASKS)
    suite_task_ids = tuple(ep.task_id for ep in suite.episodes)
    assert suite_task_ids == tuple(SOLAR_TASKS)
    seeds = [ep.seed for ep in suite.episodes]
    assert len(set(seeds)) == len(seeds), "task suite seeds must be unique for reproducibility"
    for ep in suite.episodes:
        assert ep.task_id
        assert ep.episode_id == f"task:{ep.task_id}:{ep.seed}"


def test_solar_tasks_suite_episodes_build_via_task_path():
    gen = ScenarioGenerator()
    for ep in get_suite("solar_tasks").episodes:
        world = gen.build(seed=ep.seed, scenario_family=ep.scenario_family, task_id=ep.task_id)
        spec = SOLAR_TASKS[ep.task_id]
        assert world.scenario_family == ep.scenario_family
        assert world.mission.task_id == ep.task_id
        assert world.mission.task_name == spec.name
        assert world.max_steps == spec.max_steps
        assert world.telemetry.battery.level_pct == pytest.approx(spec.initial_battery_pct)


def test_solar_tasks_suite_runs_end_to_end_with_random_policy():
    report = run_suite(RandomPolicy(seed=123), suite="solar_tasks")

    assert report.suite == "solar_tasks"
    assert report.episodes == len(SOLAR_TASKS)
    for row in report.rows:
        assert row.steps > 0
        assert isinstance(row.total_reward, float)
        assert -1.0 <= row.total_reward <= 1.0
        assert row.episode_id.startswith("task:")


def test_solar_tasks_suite_runs_end_to_end_with_scripted_policy():
    report = run_suite(ScriptedPolicy(), suite="solar_tasks")

    assert report.episodes == len(SOLAR_TASKS)
    for row in report.rows:
        assert "total" in row.reward_breakdown
        assert "safety_gate" in row.reward_breakdown
        assert "integrity_gate" in row.reward_breakdown


def test_inspect_recapture_quality_loop_requires_better_framing():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7, task="inspect_recapture_quality_loop")
    start_from_north_corridor(env)

    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-30, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="bad pitch"))

    assert obs.error is None
    assert obs.last_capture.target_quality("row_B6") < obs.mission.min_capture_quality
    assert "row_B6" not in obs.checklist_status.thermal_rows_covered

    env.step(act("fly_to_viewpoint", x=30, y=16, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="better framing"))

    assert obs.last_capture.target_quality("row_B6") >= obs.mission.min_capture_quality
    assert "hotspot_B6_quality" in obs.checklist_status.anomalies_detected


def test_zoom_required_long_standoff_crosses_rgb_threshold_only_with_zoom():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7, task="zoom_required_long_standoff")
    start_from_north_corridor(env)

    env.step(act("fly_to_viewpoint", x=60, y=20, z=16, yaw_deg=180, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=60, y=8, z=16, yaw_deg=180, speed_mps=5))
    env.step(act("set_camera_source", source="rgb"))
    env.step(act("set_gimbal", pitch_deg=-45, yaw_deg=0))
    obs = env.step(act("capture_rgb", label="far no zoom"))

    assert obs.error is None
    assert obs.last_capture.target_quality("row_B7") < obs.mission.min_rgb_quality

    env.step(act("set_zoom", zoom_level=2.0))
    obs = env.step(act("capture_rgb", label="far with zoom"))

    assert obs.last_capture.target_quality("row_B7") >= obs.mission.min_rgb_quality


def test_edge_row_quality_bar_needs_dedicated_edge_viewpoint():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7, task="edge_row_quality_bar")
    start_from_north_corridor(env)

    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="standard north overview"))

    assert obs.last_capture.target_quality("row_B8") < obs.mission.min_capture_quality
    assert obs.checklist_status.anomalies_detected == []

    env.step(act("fly_to_viewpoint", x=30, y=32, z=22, yaw_deg=-90, speed_mps=5))
    obs = env.step(act("capture_thermal", label="edge framing"))

    assert obs.last_capture.target_quality("row_B8") >= obs.mission.min_capture_quality
    assert "hotspot_B8_edge_quality" in obs.checklist_status.anomalies_detected


def test_soft_privacy_zone_allows_flight_but_blocks_capture():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="soft_privacy_capture_positioning")
    start_from_north_corridor(env)

    obs = env.step(act("fly_to_viewpoint", x=45, y=8, z=16, yaw_deg=180, speed_mps=5))
    assert obs.error is None

    obs = env.step(act("capture_rgb", label="privacy violation"))
    assert "privacy_capture_violation:soft_privacy_rgb_standoff" in obs.error
    assert obs.reward_breakdown.safety_gate == 0.0


def test_thermal_only_anomaly_is_full_issue_credit_without_rgb():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2414, task="thermal_only_anomaly_skip_rgb")
    defect = env.debug_world.hidden_defects[0]

    assert defect.requires_rgb_context is False
    assert env.debug_world.mission.rgb_closeup_for_anomalies is False


def test_strict_weighted_triage_publishes_three_different_issue_weights():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2422, task="strict_severity_weighted_triage")

    weights = sorted(defect.weight for defect in env.debug_world.hidden_defects)
    assert weights == [pytest.approx(0.5), pytest.approx(1.0), pytest.approx(3.0)]
    assert obs.telemetry.battery.level_pct == pytest.approx(52.0)
    assert env.debug_world.max_steps == 22


def test_open_items_improve_grounding_for_missing_rows():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2406, task="honest_partial_report_open_items")

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=20, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    env.step(act("capture_thermal", label="partial thermal"))

    no_open_report = {
        "summary": "Partial inspection; returned home with battery reserve.",
        "photo_ids": ["IMG-T-001"],
        "findings": [],
        "open_items": [],
        "safety_notes": ["Returned home with battery reserve."],
    }
    open_report = no_open_report | {
        "open_items": [{"target_id": "row_B4", "reason": "battery and step reserve limited completion"}]
    }

    from dronecaptureops.core.models import EvidenceReport

    no_open_score, _ = validate_evidence_report(env.debug_world, EvidenceReport(**no_open_report))
    open_score, _ = validate_evidence_report(env.debug_world, EvidenceReport(**open_report))

    assert open_score > no_open_score


def test_diode_fault_task_uses_bypass_diode_camera_branch():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2418, task="diode_fault_needs_close_thermal")
    defect = env.debug_world.hidden_defects[0]

    assert defect.defect_type == "bypass_diode_fault"
    assert defect.min_quality == pytest.approx(0.65)
