from dronecaptureops.core.constants import THERMAL_EFFECTIVE_RANGE_M, THERMAL_HFOV_DEG, THERMAL_VFOV_DEG
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction
from dronecaptureops.rich_sim import build_scene_event, build_scene_from_observation, build_scene_from_world


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_scene_from_observation_serializes_visible_renderer_state():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7)
    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    obs = env.step(act("capture_thermal", label="north overview"))

    scene = build_scene_from_observation(
        obs,
        route_history=[
            obs.telemetry.pose,
            {"pose": {"x": 30.0, "y": 24.0, "z": 22.0, "yaw_deg": -90.0}, "elapsed_time_s": 12.5},
        ],
    )
    payload = scene.model_dump(mode="json")

    assert payload["schema_version"] == "rich_sim.scene.v1"
    assert payload["metadata"]["episode_id"] == obs.metadata["episode_id"]
    assert payload["drone"]["pose"]["x"] == obs.telemetry.pose.x
    assert payload["home_pad"]["pose"] == {"x": 0.0, "y": 0.0, "z": 0.0, "yaw_deg": 0.0}
    assert len(payload["route_history"]) == 2
    assert len(payload["assets"]) == 5
    assert {zone["zone_type"] for zone in payload["airspace_zones"]} >= {"no_fly"}
    assert payload["viewpoints"], "solar scenarios should expose candidate viewpoints"
    assert payload["telemetry"]["battery_pct"] == obs.telemetry.battery.level_pct
    assert payload["checklist"]["mission_phase"] == obs.inspection_affordances.mission_phase
    assert payload["reward"]["total"] == obs.reward_breakdown.total

    capture = payload["capture_points"][0]
    assert capture["photo_id"] == "IMG-T-001"
    assert capture["frustum"]["sensor"] == "thermal"
    assert capture["frustum"]["hfov_deg"] == THERMAL_HFOV_DEG
    assert capture["frustum"]["vfov_deg"] == THERMAL_VFOV_DEG
    assert capture["frustum"]["range_m"] == THERMAL_EFFECTIVE_RANGE_M


def test_scene_from_world_uses_visible_fields_without_hidden_state_leakage():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    world = env.debug_world

    scene = build_scene_from_world(world)
    scene_json = scene.model_dump_json()

    assert world.hidden_defects, "test seed should include verifier-only defects"
    for defect in world.hidden_defects:
        assert defect.defect_id not in scene_json
        assert defect.defect_type not in scene_json

    for forbidden in [
        "hidden_defects",
        "true_asset_state",
        "hidden_weather_details",
        "obstacle_schedule",
        "verifier_evidence_requirements",
        "known_to_agent",
        "nominal",
    ]:
        assert forbidden not in scene_json

    assert scene.assets
    assert scene.airspace_zones
    assert scene.reward is not None


def test_scene_events_are_model_dump_compatible():
    event = build_scene_event(
        "camera.capture",
        {"photo_id": "IMG-T-001", "accepted": True},
        step_count=4,
        elapsed_time_s=18.0,
    )

    assert event.model_dump(mode="json") == {
        "schema_version": "rich_sim.event.v1",
        "event_type": "camera.capture",
        "step_count": 4,
        "elapsed_time_s": 18.0,
        "payload": {"photo_id": "IMG-T-001", "accepted": True},
    }
