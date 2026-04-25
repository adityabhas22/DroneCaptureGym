"""World construction helpers."""

from __future__ import annotations

from dronecaptureops.core.models import AirspaceZone
from dronecaptureops.core.state import EpisodeWorld


def mark_return_status(world: EpisodeWorld) -> None:
    """Update return-home checklist flags from current telemetry."""

    pose = world.telemetry.pose
    home = world.home_pose
    near_home = abs(pose.x - home.x) <= 1.0 and abs(pose.y - home.y) <= 1.0
    world.checklist_status.returned_home = near_home
    world.checklist_status.landed = world.telemetry.landed
    world.telemetry.sync_legacy_fields()


def is_zone_active(zone: AirspaceZone, world: EpisodeWorld) -> bool:
    """Return True if the zone is in effect right now.

    Zones default to always-on. A zone whose ID appears in
    `world.obstacle_schedule` is on only while the current step lies in
    `[active_from_step, active_until_step]`. This is what makes
    `blocked_corridor_replan` scenarios actually replannable: once the
    crane window closes, the corresponding obstacle is no longer applied.
    """

    schedule = next(
        (entry for entry in world.obstacle_schedule if entry.get("zone_id") == zone.zone_id),
        None,
    )
    if schedule is None:
        return True
    start = int(schedule.get("active_from_step", 0))
    end_value = schedule.get("active_until_step")
    end = int(end_value) if end_value is not None else 10**9
    return start <= world.step_count <= end


def active_zones(world: EpisodeWorld) -> list[AirspaceZone]:
    """Return zones currently in effect (filtered by obstacle_schedule)."""

    return [zone for zone in world.airspace_zones if is_zone_active(zone, world)]
