from dronecaptureops.generation.scenario_generator import ScenarioGenerator


def test_solar_scenario_contains_required_rows_and_no_fly_zone():
    scenario = ScenarioGenerator().build(seed=5, domain="solar")

    assert scenario.mission.required_rows == ["row_B4", "row_B5", "row_B6", "row_B7", "row_B8"]
    assert any(zone.zone_type == "no_fly" for zone in scenario.airspace_zones)
    assert scenario.hidden_defects


def test_solar_scenario_populates_generic_assets_zones_and_viewpoints():
    scenario = ScenarioGenerator().build(seed=7, domain="solar")

    assert [asset.asset_id for asset in scenario.assets] == ["row_B4", "row_B5", "row_B6", "row_B7", "row_B8"]
    assert all(asset.asset_type == "solar_row" for asset in scenario.assets)
    assert all(asset.safe_standoff_bands for asset in scenario.assets)
    assert scenario.airspace_zones[0].constraint_level == "hard"
    assert {viewpoint.viewpoint_id for viewpoint in scenario.viewpoints} == {
        "vp_block_b_west_overview",
        "vp_block_b_close_context",
    }
