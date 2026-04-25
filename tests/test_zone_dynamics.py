"""Obstacle-schedule activation and request_route_replan filtering."""

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_request_route_replan_filters_viewpoints_inside_active_hard_zones():
    """When a hard obstacle covers a viewpoint pose, the replan tool drops it."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2301, scenario_family="blocked_corridor_replan")

    obs = env.step(act("request_route_replan", reason="crane corridor blocks close-up route"))
    blocked_ids = obs.action_result["blocked_zone_ids"]
    recommended = obs.action_result["recommended_viewpoints"]

    assert "temporary_crane_corridor" in blocked_ids

    # No recommended viewpoint should sit inside the temporary crane zone.
    crane = next(zone for zone in env.debug_world.airspace_zones if zone.zone_id == "temporary_crane_corridor")
    for viewpoint in recommended:
        pose = viewpoint["pose"]
        in_zone = (
            crane.min_x <= pose["x"] <= crane.max_x
            and crane.min_y <= pose["y"] <= crane.max_y
            and crane.min_altitude_m <= pose["z"] <= crane.max_altitude_m
        )
        assert not in_zone, f"viewpoint {viewpoint['viewpoint_id']} sits inside the active crane zone"


def test_obstacle_schedule_deactivates_after_window():
    """A zone with active_until_step is no longer enforced after that step."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2301, scenario_family="blocked_corridor_replan")

    schedule = env.debug_world.obstacle_schedule
    assert schedule, "blocked_corridor_replan should ship an obstacle_schedule"
    crane_window = next(entry for entry in schedule if entry["zone_id"] == "temporary_crane_corridor")

    # Force step_count past the activation window.
    env.debug_world.step_count = crane_window["active_until_step"] + 5

    obs = env.step(act("request_route_replan", reason="check after window"))
    assert "temporary_crane_corridor" not in obs.action_result["blocked_zone_ids"]


def test_safety_check_skips_inactive_obstacle_zones():
    """Once the obstacle window expires, a path through it is allowed."""

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2301, scenario_family="blocked_corridor_replan")
    env.step(act("takeoff", altitude_m=18))

    # Advance past the obstacle window.
    schedule = env.debug_world.obstacle_schedule
    crane_window = next(entry for entry in schedule if entry["zone_id"] == "temporary_crane_corridor")
    env.debug_world.step_count = crane_window["active_until_step"] + 1

    # A waypoint that previously sat inside the crane zone should now be reachable.
    crane = next(zone for zone in env.debug_world.airspace_zones if zone.zone_id == "temporary_crane_corridor")
    inside_x = (crane.min_x + crane.max_x) / 2
    inside_y = (crane.min_y + crane.max_y) / 2
    obs = env.step(
        act("fly_to_viewpoint", x=inside_x, y=inside_y, z=20, yaw_deg=0, speed_mps=5)
    )
    # The fly succeeds (no obstacle violation) once the schedule window has closed.
    assert obs.error is None or "obstacle_violation" not in obs.error
