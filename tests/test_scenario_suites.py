from dronecaptureops.generation.scenario_generator import ScenarioGenerator
from dronecaptureops.generation.suites import get_suite, list_suites, resolve_suite_episodes


def test_scenario_suites_are_centralized_and_deterministic():
    suite = get_suite("demo")

    assert suite.episodes[0].episode_id == "single_hotspot:2101"
    assert suite.families == ("single_hotspot", "false_positive_glare", "blocked_corridor_replan")
    assert any(item.name == "hard_eval" for item in list_suites())


def test_live_llm_demo_suite_is_task_conditioned():
    suite = get_suite("demo_llm_inspection")

    assert suite.episodes[0].episode_id == "task:basic_thermal_survey:2501"
    assert suite.families == (
        "single_hotspot",
        "blocked_corridor_replan",
        "low_battery_tradeoff",
        "false_positive_glare",
    )
    assert all("rich_sim" in episode.tags for episode in suite.episodes)
    assert all(episode.task_id for episode in suite.episodes)


def test_resolved_suite_episode_builds_matching_scenario_family():
    episode = resolve_suite_episodes(suite_name="hard_eval")[0]
    scenario = ScenarioGenerator().build(
        seed=episode.seed,
        domain="solar",
        scenario_family=episode.scenario_family,
    )

    assert scenario.scenario_family == episode.scenario_family
    assert scenario.mission.difficulty == "hard"
    assert any(zone.zone_id == "temporary_crane_corridor" for zone in scenario.airspace_zones)
    assert scenario.obstacle_schedule


def test_medium_false_positive_glare_has_visible_operational_constraints():
    scenario = ScenarioGenerator().build(seed=2203, domain="solar", scenario_family="false_positive_glare")

    assert scenario.mission.scenario_family == "false_positive_glare"
    assert "glare_risk:high" in scenario.mission.environmental_constraints
    assert scenario.weather.irradiance_wm2 >= 900
    assert any(zone.zone_type == "privacy" for zone in scenario.airspace_zones)
