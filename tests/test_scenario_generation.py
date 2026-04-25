from dronecaptureops.generation.scenario_generator import ScenarioGenerator


def test_solar_scenario_contains_required_rows_and_no_fly_zone():
    scenario = ScenarioGenerator().build(seed=5, domain="solar")

    assert scenario.mission.required_rows == ["row_B4", "row_B5", "row_B6", "row_B7", "row_B8"]
    assert any(zone.zone_type == "no_fly" for zone in scenario.restricted_zones)
    assert scenario.hidden_defects
