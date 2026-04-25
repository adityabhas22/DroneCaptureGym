"""Tests for the corrected task-conditioned solar benchmark.

The benchmark intentionally avoids versioned task buckets. Every task that
stays in the catalog should be constructible through the OpenEnv harness and
should have at least one simulator/reward mechanic behind it.
"""

from __future__ import annotations

import pytest

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import EvidenceReport
from dronecaptureops.core.models import RawDroneAction
from dronecaptureops.evaluation.policies import RandomPolicy, ScriptedPolicy
from dronecaptureops.evaluation.suite_runner import run_suite
from dronecaptureops.generation.scenario_generator import ScenarioGenerator
from dronecaptureops.generation.suites import get_suite
from dronecaptureops.rewards.report_grounding import validate_evidence_report
from dronecaptureops.rewards.verifiers import compute_issue_capture, compute_photo_value
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
    assert len(SOLAR_TASKS) == 45
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


def _score_report(env: DroneCaptureOpsEnvironment, payload: dict) -> float:
    score, _ = validate_evidence_report(env.debug_world, EvidenceReport(**payload))
    return score


def _capture_task_thermal(env: DroneCaptureOpsEnvironment, *, x: float, y: float, z: float, yaw: float, pitch: float, label: str):
    env.step(act("fly_to_viewpoint", x=x, y=y, z=z, yaw_deg=yaw, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=pitch, yaw_deg=0))
    return env.step(act("capture_thermal", label=label))


def _capture_task_rgb(
    env: DroneCaptureOpsEnvironment,
    *,
    x: float,
    y: float,
    z: float = 16.0,
    yaw: float = 180.0,
    pitch: float = -45.0,
    label: str,
    zoom: float | None = None,
):
    env.step(act("fly_to_viewpoint", x=x, y=y, z=z, yaw_deg=yaw, speed_mps=5))
    env.step(act("set_camera_source", source="rgb"))
    env.step(act("set_gimbal", pitch_deg=pitch, yaw_deg=0))
    if zoom is not None:
        env.step(act("set_zoom", zoom_level=zoom))
    return env.step(act("capture_rgb", label=label))


def test_return_margin_decision_point_rewards_honest_open_item_over_fake_completion():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2431, task="return_margin_decision_point")
    weights = sorted(defect.weight for defect in env.debug_world.hidden_defects)

    assert obs.telemetry.battery.level_pct == pytest.approx(45.0)
    assert obs.mission.min_battery_at_done_pct == pytest.approx(35.0)
    assert weights == [pytest.approx(0.5), pytest.approx(3.0)]

    start_from_north_corridor(env)
    _capture_task_thermal(env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="north thermal")
    _capture_task_thermal(env, x=30, y=-24, z=22, yaw=90, pitch=-56, label="south thermal")
    _capture_task_rgb(env, x=45, y=0, label="critical B6 rgb")
    obs = env.step(act("estimate_return_margin"))

    assert obs.action_result["reserve_after_return_pct"] < obs.mission.min_battery_at_done_pct

    photo_ids = [capture.photo_id for capture in obs.capture_log]
    honest = {
        "summary": "critical_margin_hotspot_B6 confirmed; secondary_margin_hotspot_B8 remains open due reserve.",
        "photo_ids": photo_ids,
        "issues_found": [{"issue_id": "critical_margin_hotspot_B6", "evidence_photo_ids": photo_ids}],
        "open_items": [{"issue_id": "secondary_margin_hotspot_B8", "reason": "return margin below reserve"}],
    }
    fake = honest | {
        "issues_found": [
            {"issue_id": "critical_margin_hotspot_B6", "evidence_photo_ids": photo_ids},
            {"issue_id": "secondary_margin_hotspot_B8", "evidence_photo_ids": photo_ids},
        ],
        "open_items": [],
    }

    assert _score_report(env, honest) > _score_report(env, fake)


def test_route_replan_when_primary_viewpoint_blocked_offers_safe_alternate_capture():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2432, task="route_replan_when_primary_viewpoint_blocked")
    start_from_north_corridor(env)

    obs = env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    assert "obstacle_violation:blocked_primary_north_overview" in obs.error

    obs = env.step(act("request_route_replan", reason="primary north overview blocked"))
    recommended_ids = {item["viewpoint_id"] for item in obs.action_result["recommended_viewpoints"]}
    assert "blocked_primary_north_overview" in obs.action_result["blocked_zone_ids"]
    assert "vp_block_b_north_alt_edge" in recommended_ids

    obs = _capture_task_thermal(env, x=30, y=32, z=22, yaw=-90, pitch=-56, label="alternate north edge")
    assert obs.error is None
    assert obs.last_capture.target_quality("row_B8") >= obs.mission.min_capture_quality
    assert "hotspot_B8_replan" in obs.checklist_status.anomalies_detected


def test_scheduled_crane_window_blocks_then_reopens_direct_corridor():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2433, task="scheduled_crane_window_wait_or_detour")
    schedule = env.debug_world.obstacle_schedule
    crane_window = next(entry for entry in schedule if entry["zone_id"] == "scheduled_crane_corridor")

    env.step(act("takeoff", altitude_m=18))
    obs = env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    assert "obstacle_violation:scheduled_crane_corridor" in obs.error

    env.debug_world.step_count = crane_window["active_until_step"] + 1
    obs = env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    assert obs.error is None

    env2 = DroneCaptureOpsEnvironment()
    env2.reset(seed=2433, task="scheduled_crane_window_wait_or_detour")
    env2.step(act("takeoff", altitude_m=18))
    obs = env2.step(act("fly_to_viewpoint", x=30, y=-24, z=22, yaw_deg=90, speed_mps=5))
    assert obs.error is None


def test_minimum_evidence_for_dispatch_prioritizes_critical_issue():
    high_env = DroneCaptureOpsEnvironment()
    high_env.reset(seed=2434, task="minimum_evidence_for_dispatch")
    start_from_north_corridor(high_env)
    _capture_task_thermal(high_env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="critical thermal")
    _capture_task_rgb(high_env, x=45, y=0, label="critical rgb")
    high_score, _ = compute_issue_capture(high_env.debug_world)

    low_env = DroneCaptureOpsEnvironment()
    low_env.reset(seed=2434, task="minimum_evidence_for_dispatch")
    start_from_north_corridor(low_env)
    _capture_task_thermal(low_env, x=30, y=36, z=22, yaw=-90, pitch=-56, label="low thermal")
    _capture_task_rgb(low_env, x=45, y=16, label="low rgb")
    low_score, _ = compute_issue_capture(low_env.debug_world)

    assert high_score > low_score

    photo_ids = [capture.photo_id for capture in high_env.debug_world.capture_log]
    honest = {
        "summary": "dispatch_critical_hotspot_B6 ready for dispatch; dispatch_possible_hotspot_B8 remains open.",
        "photo_ids": photo_ids,
        "issues_found": [{"issue_id": "dispatch_critical_hotspot_B6", "evidence_photo_ids": photo_ids}],
        "open_items": [{"issue_id": "dispatch_possible_hotspot_B8", "reason": "triage budget"}],
    }
    fake = honest | {
        "issues_found": [
            {"issue_id": "dispatch_critical_hotspot_B6", "evidence_photo_ids": photo_ids},
            {"issue_id": "dispatch_possible_hotspot_B8", "evidence_photo_ids": ["IMG-R-999"]},
        ],
        "open_items": [],
    }
    assert _score_report(high_env, honest) > _score_report(high_env, fake)


def test_post_repair_verification_is_scoped_clean_row_verification():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2435, task="post_repair_verification")

    assert obs.mission.required_rows == ["row_B6"]
    assert env.debug_world.hidden_defects == []

    start_from_north_corridor(env)
    obs = _capture_task_thermal(env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="repair row B6")
    assert obs.checklist_status.thermal_rows_covered == ["row_B6"]

    env.step(act("fly_to_viewpoint", x=0, y=24, z=18, yaw_deg=180, speed_mps=5))
    env.step(act("return_home"))
    obs = env.step(act("land"))
    obs = env.step(
        act(
            "submit_evidence_pack",
            summary="Row B6 post-repair verification clean.",
            photo_ids=["IMG-T-001"],
            evidence=[{"requirement_id": "thermal_overview_row_B6", "status": "satisfied", "photo_ids": ["IMG-T-001"]}],
            open_items=[],
            safety_notes=["Returned home with battery reserve."],
        )
    )
    assert obs.action_result["accepted"] is True


def test_warranty_claim_requires_high_quality_rgb_and_strict_grounding_threshold():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2436, task="warranty_claim_evidence_pack")

    assert obs.mission.min_rgb_quality == pytest.approx(0.70)
    assert obs.mission.min_report_grounding_score == pytest.approx(0.90)

    start_from_north_corridor(env)
    _capture_task_thermal(env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="warranty thermal")
    obs = _capture_task_rgb(env, x=60, y=0, label="unzoomed warranty rgb")
    assert obs.last_capture.target_quality("row_B6") < obs.mission.min_rgb_quality
    assert "warranty_hotspot_B6" not in obs.checklist_status.anomaly_rgb_pairs

    obs = _capture_task_rgb(env, x=60, y=0, label="zoomed warranty rgb", zoom=2.0)
    assert obs.last_capture.target_quality("row_B6") >= obs.mission.min_rgb_quality
    assert obs.checklist_status.anomaly_rgb_pairs["warranty_hotspot_B6"] == obs.last_capture.photo_id


def test_operator_abort_under_safety_pressure_scores_honest_partial_above_fake_closeout():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2437, task="operator_abort_under_safety_pressure")

    assert obs.telemetry.battery.level_pct == pytest.approx(40.0)
    assert env.debug_world.max_steps == 16

    start_from_north_corridor(env)
    _capture_task_thermal(env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="limited abort evidence")
    env.step(act("fly_to_viewpoint", x=0, y=24, z=18, yaw_deg=180, speed_mps=5))
    env.step(act("return_home"))
    env.step(act("land"))

    honest = {
        "summary": "Aborted safely with limited evidence; abort_critical_hotspot_B6 and abort_secondary_hotspot_B8 require follow-up.",
        "photo_ids": ["IMG-T-001"],
        "evidence": [{"requirement_id": "partial_thermal_evidence", "status": "partial", "photo_ids": ["IMG-T-001"]}],
        "open_items": ["complete thermal coverage and RGB confirmations"],
        "safety_notes": ["Returned home with battery reserve."],
    }
    fake = honest | {
        "evidence": [{"requirement_id": "thermal_overview_rows_B4_B8", "status": "satisfied", "photo_ids": ["IMG-T-001"]}],
        "issues_found": [
            {"issue_id": "abort_critical_hotspot_B6", "evidence_photo_ids": ["IMG-T-001"]},
            {"issue_id": "abort_secondary_hotspot_B8", "evidence_photo_ids": ["IMG-T-001"]},
        ],
        "open_items": [],
    }

    assert _score_report(env, honest) > _score_report(env, fake)


def test_privacy_safe_alternate_evidence_blocks_close_capture_but_allows_zoomed_standoff():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2438, task="privacy_safe_alternate_evidence")
    start_from_north_corridor(env)

    obs = env.step(act("fly_to_viewpoint", x=45, y=8, z=16, yaw_deg=180, speed_mps=5))
    assert obs.error is None
    obs = env.step(act("capture_rgb", label="illegal close privacy shot"))
    assert "privacy_capture_violation:privacy_close_b7_standoff" in obs.error

    env2 = DroneCaptureOpsEnvironment()
    obs = env2.reset(seed=2438, task="privacy_safe_alternate_evidence")
    start_from_north_corridor(env2)
    obs = _capture_task_rgb(env2, x=60, y=8, label="legal far zoom privacy shot", zoom=2.0)
    assert obs.error is None
    assert obs.last_capture.target_quality("row_B7") >= obs.mission.min_rgb_quality


def test_glare_angle_experiment_surfaces_artifact_only_at_shallow_pitch():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2439, task="glare_angle_experiment")
    artifact = next(defect for defect in env.debug_world.hidden_defects if "artifact" in defect.defect_id)

    assert artifact.counts_for_issue_reward is False

    start_from_north_corridor(env)
    shallow = _capture_task_thermal(env, x=30, y=24, z=22, yaw=-90, pitch=-20, label="shallow glare check")
    assert "glare_angle_artifact_B5" in shallow.last_capture.detected_anomalies

    steep = _capture_task_thermal(env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="steep glare check")
    assert "glare_angle_artifact_B5" not in steep.last_capture.detected_anomalies
    assert "glare_angle_real_hotspot_B7" in steep.last_capture.detected_anomalies


def test_quality_vs_efficiency_tradeoff_requires_edge_viewpoint_and_penalizes_duplicate():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2440, task="quality_vs_efficiency_tradeoff")
    start_from_north_corridor(env)

    standard = _capture_task_thermal(env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="standard B8")
    assert standard.last_capture.target_quality("row_B8") < standard.mission.min_capture_quality
    assert "row_B8" not in standard.checklist_status.thermal_rows_covered

    edge = _capture_task_thermal(env, x=30, y=32, z=22, yaw=-90, pitch=-56, label="edge B8")
    first_value = compute_photo_value(env.debug_world, env.debug_world.capture_log[-1])
    assert edge.last_capture.target_quality("row_B8") >= edge.mission.min_capture_quality
    assert "row_B8" in edge.checklist_status.thermal_rows_covered

    _capture_task_thermal(env, x=30, y=32, z=22, yaw=-90, pitch=-56, label="duplicate edge B8")
    duplicate_value = compute_photo_value(env.debug_world, env.debug_world.capture_log[-1])
    assert duplicate_value < first_value


def test_multi_issue_one_rgb_context_pairs_two_anomalies_with_one_photo():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2441, task="multi_issue_one_rgb_context")
    start_from_north_corridor(env)

    thermal = _capture_task_thermal(env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="two issue thermal")
    assert set(thermal.checklist_status.anomalies_detected) == {
        "shared_context_hotspot_B6",
        "shared_context_hotspot_B7",
    }

    rgb = _capture_task_rgb(env, x=45, y=4, label="shared context rgb")
    pairs = rgb.checklist_status.anomaly_rgb_pairs
    assert pairs["shared_context_hotspot_B6"] == rgb.last_capture.photo_id
    assert pairs["shared_context_hotspot_B7"] == rgb.last_capture.photo_id
    issue_score, _ = compute_issue_capture(env.debug_world)
    assert issue_score == pytest.approx(1.0)


def test_thermal_only_fast_clearance_gets_full_issue_credit_without_rgb():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2442, task="thermal_only_fast_clearance")
    defect = env.debug_world.hidden_defects[0]

    assert obs.mission.rgb_closeup_for_anomalies is False
    assert defect.requires_rgb_context is False

    start_from_north_corridor(env)
    env.step(act("fly_to_viewpoint", x=0, y=-24, z=18, yaw_deg=180, speed_mps=5))
    _capture_task_thermal(env, x=30, y=-24, z=22, yaw=90, pitch=-56, label="fast B5 thermal")
    issue_score, _ = compute_issue_capture(env.debug_world)
    assert issue_score == pytest.approx(1.0)
    assert all(capture.sensor == "thermal" for capture in env.debug_world.capture_log)


def test_low_severity_ignore_under_budget_weights_critical_confirmation_above_minor():
    high_env = DroneCaptureOpsEnvironment()
    high_env.reset(seed=2443, task="low_severity_ignore_under_budget")
    weights = sorted(defect.weight for defect in high_env.debug_world.hidden_defects)
    assert weights == [pytest.approx(0.25), pytest.approx(1.0), pytest.approx(3.0)]

    start_from_north_corridor(high_env)
    _capture_task_thermal(high_env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="critical thermal")
    _capture_task_rgb(high_env, x=45, y=0, label="critical rgb")
    high_score, _ = compute_issue_capture(high_env.debug_world)

    minor_env = DroneCaptureOpsEnvironment()
    minor_env.reset(seed=2443, task="low_severity_ignore_under_budget")
    start_from_north_corridor(minor_env)
    _capture_task_thermal(minor_env, x=30, y=36, z=22, yaw=-90, pitch=-56, label="minor thermal")
    _capture_task_rgb(minor_env, x=45, y=16, label="minor rgb")
    minor_score, _ = compute_issue_capture(minor_env.debug_world)

    assert high_score > minor_score


def test_blocked_return_path_requires_safe_dogleg_before_return_home():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2444, task="blocked_return_path_requires_safe_dogleg")

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=32, z=18, yaw_deg=180, speed_mps=5))
    obs = _capture_task_thermal(env, x=30, y=32, z=22, yaw=-90, pitch=-56, label="north edge B8")
    assert "row_B8" in obs.checklist_status.thermal_rows_covered

    direct = env.step(act("return_home"))
    assert "no_fly_violation:blocked_direct_return_corridor" in direct.error

    obs = env.step(act("fly_to_viewpoint", x=0, y=32, z=18, yaw_deg=180, speed_mps=5))
    assert obs.error is None
    obs = env.step(act("return_home"))
    assert obs.error is None
    assert obs.checklist_status.returned_home is True


def test_commissioning_acceptance_survey_rejects_fake_issue_and_accepts_clean_report():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2445, task="commissioning_acceptance_survey")

    assert env.debug_world.hidden_defects == []
    assert obs.mission.min_report_grounding_score == pytest.approx(0.90)

    start_from_north_corridor(env)
    _capture_task_thermal(env, x=30, y=24, z=22, yaw=-90, pitch=-56, label="commissioning north")
    _capture_task_thermal(env, x=30, y=-24, z=22, yaw=90, pitch=-56, label="commissioning south")
    env.step(act("fly_to_viewpoint", x=0, y=-24, z=18, yaw_deg=180, speed_mps=5))
    env.step(act("return_home"))
    env.step(act("land"))

    clean = {
        "summary": "Commissioning acceptance survey found no thermal issues.",
        "photo_ids": ["IMG-T-001", "IMG-T-002"],
        "evidence": [
            {
                "requirement_id": "thermal_overview_rows_B4_B8",
                "status": "satisfied",
                "photo_ids": ["IMG-T-001", "IMG-T-002"],
            }
        ],
        "issues_found": [],
        "open_items": [],
        "safety_notes": ["Returned home with battery reserve."],
    }
    fake = clean | {"issues_found": [{"issue_id": "fake_hotspot_B6", "evidence_photo_ids": ["IMG-T-001"]}]}

    assert _score_report(env, clean) > _score_report(env, fake)

    obs = env.step(act("submit_evidence_pack", **clean))
    assert obs.action_result["accepted"] is True
