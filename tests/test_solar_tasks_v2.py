"""Tests for the v2 task-conditioned solar mission catalog.

The v2 catalog (15 tasks) overlays new mission variants onto the existing
solar scenario families. These tests pin the contracts that matter for RL
training and the OpenEnv harness:

* every new task constructs a deterministic ``EpisodeWorld`` via the
  task-conditioned reset path (no exceptions, no missing fields);
* hidden state stays hidden (defects + true asset state never appear in the
  agent-visible observation or visible state);
* the ``solar_tasks_v2`` suite resolves to one episode per task and round-
  trips through the suite runner with both random and scripted policies;
* the most distinctive new behaviors are pinned (permanent occlusion blocks
  the north overview, glare-only artifacts do not score as real issues, and
  the audit-grade task surfaces the stricter grounding bar).
"""

from __future__ import annotations

import pytest

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction
from dronecaptureops.evaluation.policies import RandomPolicy, ScriptedPolicy
from dronecaptureops.evaluation.suite_runner import run_suite
from dronecaptureops.generation.scenario_generator import ScenarioGenerator
from dronecaptureops.generation.suites import get_suite
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


V2_TASK_IDS: tuple[str, ...] = (
    "string_outage_survey",
    "pid_multi_row_pattern",
    "cracked_glass_closeup",
    "bird_soiling_explanation",
    "vegetation_edge_encroachment",
    "substation_adjacency_caution",
    "low_contrast_recapture",
    "true_false_anomaly_discrimination",
    "permanent_occlusion_coverage",
    "prioritized_triage_under_constraint",
    "capture_efficiency_discipline",
    "boundary_aware_closeup",
    "no_defect_with_glare_artifact",
    "adaptive_battery_reserve",
    "audit_grade_strict_grounding",
)


def act(tool_name: str, **arguments) -> RawDroneAction:
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_v2_task_ids_are_registered_in_solar_tasks_catalog():
    assert set(V2_TASK_IDS) <= set(SOLAR_TASKS)
    assert len(V2_TASK_IDS) == 15


@pytest.mark.parametrize("task_id", V2_TASK_IDS)
def test_v2_task_resets_into_a_well_formed_observation(task_id: str):
    """Every v2 task resets cleanly and surfaces the task spec to the agent."""

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


@pytest.mark.parametrize("task_id", V2_TASK_IDS)
def test_v2_task_battery_and_zones_match_the_spec(task_id: str):
    """Battery, weather, and extra zones must come from the deterministic spec."""

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


@pytest.mark.parametrize("task_id", V2_TASK_IDS)
def test_v2_task_observation_does_not_leak_hidden_defects(task_id: str):
    """Verifier-only defect identifiers must never appear in agent-visible state.

    Mission text and task tags are intentionally public, so we check that the
    hidden defect ID does not appear as a substring outside the spec's
    intentional public hints.
    """

    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7, task=task_id)
    obs_json = obs.model_dump_json()
    state_json = env.state.model_dump_json()

    for defect in env.debug_world.hidden_defects:
        assert defect.defect_id not in obs_json, f"defect_id {defect.defect_id} leaked in obs"
        assert defect.defect_id not in state_json, f"defect_id {defect.defect_id} leaked in visible state"


def test_solar_tasks_v2_suite_has_one_episode_per_v2_task():
    suite = get_suite("solar_tasks_v2")

    assert len(suite.episodes) == len(V2_TASK_IDS)
    suite_task_ids = tuple(ep.task_id for ep in suite.episodes)
    assert suite_task_ids == V2_TASK_IDS
    seeds = [ep.seed for ep in suite.episodes]
    assert len(set(seeds)) == len(seeds), "v2 suite seeds must be unique for reproducibility"
    for ep in suite.episodes:
        assert ep.task_id, "v2 suite episodes must be task-conditioned"
        assert ep.episode_id == f"task:{ep.task_id}:{ep.seed}"


def test_solar_tasks_v2_suite_episodes_build_via_task_path():
    """Each suite episode should build through the task-conditioned overlay."""

    gen = ScenarioGenerator()
    for ep in get_suite("solar_tasks_v2").episodes:
        world = gen.build(seed=ep.seed, scenario_family=ep.scenario_family, task_id=ep.task_id)
        spec = SOLAR_TASKS[ep.task_id]
        assert world.scenario_family == ep.scenario_family
        assert world.mission.task_id == ep.task_id
        assert world.mission.task_name == spec.name
        assert world.max_steps == spec.max_steps
        assert world.telemetry.battery.level_pct == pytest.approx(spec.initial_battery_pct)


def test_solar_tasks_v2_suite_runs_end_to_end_with_random_policy():
    """A short random rollout over every v2 episode must complete cleanly."""

    report = run_suite(RandomPolicy(seed=123), suite="solar_tasks_v2")

    assert report.suite == "solar_tasks_v2"
    assert report.episodes == len(V2_TASK_IDS)
    for row in report.rows:
        assert row.steps > 0
        assert isinstance(row.total_reward, float)
        assert -1.0 <= row.total_reward <= 1.0
        assert row.episode_id.startswith("task:")


def test_solar_tasks_v2_suite_runs_end_to_end_with_scripted_policy():
    """Scripted policy must also drive each v2 episode without raising.

    We do NOT assert success per-episode here: the v2 catalog is intentionally
    harder than the v1 generic solver path (e.g. permanent occlusion needs
    task-aware routing). What we DO assert is that the harness, observations,
    rewards, and grounding pipeline survive the new tasks end-to-end and
    return well-formed reward breakdowns.
    """

    report = run_suite(ScriptedPolicy(), suite="solar_tasks_v2")

    assert report.episodes == len(V2_TASK_IDS)
    for row in report.rows:
        assert "total" in row.reward_breakdown
        assert "safety_gate" in row.reward_breakdown
        assert "integrity_gate" in row.reward_breakdown


def test_permanent_occlusion_coverage_blocks_the_north_overview_viewpoint():
    """The maintenance vehicle obstacle must reject the standard north overview."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2409, task="permanent_occlusion_coverage")

    obs = env.step(act("takeoff", altitude_m=18))
    assert obs.error is None
    obs = env.step(act("fly_to_viewpoint", x=0, y=20, z=18, yaw_deg=0, speed_mps=5))
    assert obs.error is None

    obs = env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))

    assert obs.error is not None
    assert "obstacle_violation" in obs.error or "no_fly_violation" in obs.error


def test_no_defect_with_glare_artifact_does_not_surface_anomaly_at_steep_pitch():
    """At steep pitch the glare-only artifact must not appear in the checklist."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2413, task="no_defect_with_glare_artifact")

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=20, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="steep north overview"))

    assert obs.error is None
    assert obs.checklist_status.anomalies_detected == []


def test_true_false_discrimination_glare_artifact_is_excluded_from_issue_reward():
    """The glare-only defect must be present but flagged as non-scoring."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2408, task="true_false_anomaly_discrimination")

    real = [d for d in env.debug_world.hidden_defects if d.defect_type == "thermal_hotspot"]
    fake = [d for d in env.debug_world.hidden_defects if d.defect_type == "false_thermal_artifact"]

    assert len(real) == 1
    assert len(fake) == 1
    assert real[0].counts_for_issue_reward is True
    assert real[0].requires_rgb_context is True
    assert fake[0].counts_for_issue_reward is False
    assert fake[0].requires_rgb_context is False
    assert fake[0].weight == pytest.approx(0.0)


def test_audit_grade_task_surfaces_strict_grounding_threshold():
    """The audit-grade task must publish the stricter grounding bar to the agent."""

    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2415, task="audit_grade_strict_grounding")

    assert obs.mission.min_report_grounding_score == pytest.approx(0.85)
    assert "audit" in obs.mission.task_tags


def test_prioritized_triage_under_constraint_publishes_tight_battery_and_step_budget():
    """Triage scenario must surface tight budgets (battery + steps)."""

    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2410, task="prioritized_triage_under_constraint")

    assert obs.telemetry.battery.level_pct == pytest.approx(50.0)
    assert env.debug_world.max_steps == 24
    assert obs.mission.min_battery_at_done_pct == pytest.approx(30.0)
    severities = sorted(d.severity for d in env.debug_world.hidden_defects)
    weights = sorted(d.weight for d in env.debug_world.hidden_defects)
    assert len(severities) == 2
    assert weights == [pytest.approx(1.0), pytest.approx(3.0)]


def test_capture_efficiency_discipline_has_no_anomalies_and_tight_step_budget():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2411, task="capture_efficiency_discipline")

    assert env.debug_world.hidden_defects == []
    assert env.debug_world.max_steps == 16
    assert env.debug_world.mission.rgb_closeup_for_anomalies is False
