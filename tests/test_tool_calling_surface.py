from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_tool_catalog_and_affordances_are_visible_only_and_stateful():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=2101, scenario_family="single_hotspot")

    tool_names = {tool["name"] for tool in obs.tool_catalog}
    assert {"list_assets", "move_to_asset", "point_camera_at", "estimate_return_margin"} <= tool_names
    assert obs.inspection_affordances.mission_phase == "preflight"
    assert obs.inspection_affordances.action_availability["takeoff"] is True
    assert obs.inspection_affordances.action_availability["capture_thermal"] is False
    assert "takeoff" in obs.inspection_affordances.suggested_tools


def test_asset_level_tools_drive_visible_inspection_workflow():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2101, scenario_family="single_hotspot")

    asset_obs = env.step(act("list_assets"))
    assert asset_obs.action_result["pending_asset_ids"] == ["row_B4", "row_B5", "row_B6", "row_B7", "row_B8"]

    env.step(act("takeoff", altitude_m=18))
    moved = env.step(act("move_to_asset", asset_id="row_B6", standoff_bucket="far", speed_mps=5))
    assert moved.action_result["asset_id"] == "row_B6"
    assert moved.action_result["viewpoint_id"] in {"vp_block_b_north_overview", "vp_block_b_south_overview"}

    pointed = env.step(act("point_camera_at", asset_id="row_B6"))
    assert pointed.telemetry.gimbal.frame_mode == "roi"
    assert pointed.telemetry.gimbal.target_asset_id == "row_B6"

    source = env.step(act("set_camera_source", source="thermal"))
    assert source.telemetry.camera.active_source == "thermal"

    captured = env.step(act("capture_thermal", label="asset-level overview"))
    assert captured.last_capture is not None
    assert "row_B6" in captured.last_capture.asset_ids

    margin = env.step(act("estimate_return_margin"))
    assert margin.action_result["distance_home_m"] > 0
    assert "meets_required_reserve" in margin.action_result


def test_tool_catalog_exposes_typed_arg_schema():
    """The v2 tool catalog exposes required/optional argument names per tool.

    The legacy ArgSpec (with type/range/choices) was removed when the v2
    tool surface landed; per-arg type/range/choice validation now lives in
    the individual handlers, not in the registry catalog. The catalog keeps
    enough information for an agent to know which arguments to send.
    """

    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7)
    catalog = {tool["name"]: tool for tool in obs.tool_catalog}

    takeoff = catalog["takeoff"]
    assert "altitude_m" in takeoff["required_args"] or "altitude_m" in takeoff["optional_args"]

    set_source = catalog["set_camera_source"]
    assert "source" in set_source["required_args"] or "source" in set_source["optional_args"]

    move = catalog["move_to_asset"]
    assert "asset_id" in move["required_args"]
    assert "standoff_bucket" in move["required_args"] or "standoff_bucket" in move["optional_args"]


def test_typed_validation_rejects_wrong_types_and_unknown_enums():
    """Issue #2: registry validation surfaces a clear error before the handler runs."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    bad_type = env.step(act("takeoff", altitude_m="high"))
    assert bad_type.error and "altitude_m" in bad_type.error and "number" in bad_type.error

    bad_enum = env.step(act("set_camera_source", source="lidar"))
    assert bad_enum.error and "source" in bad_enum.error

    bad_range = env.step(act("takeoff", altitude_m=500.0))
    assert bad_range.error and "unsafe_altitude" in bad_range.error


def test_route_replan_exposes_visible_constraints_without_hidden_truth():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2301, scenario_family="blocked_corridor_replan")

    obs = env.step(act("request_route_replan", reason="temporary obstacle blocks close-up route"))

    assert "temporary_crane_corridor" in obs.action_result["blocked_zone_ids"]
    assert obs.action_result["recommended_viewpoints"]
    result_text = str(obs.action_result).lower()
    assert "hidden" not in result_text
    assert "verifier" not in result_text
