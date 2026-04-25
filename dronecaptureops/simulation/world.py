"""World construction helpers."""

from __future__ import annotations

from dronecaptureops.core.state import EpisodeWorld


def mark_return_status(world: EpisodeWorld) -> None:
    """Update return-home checklist flags from current telemetry."""

    pose = world.telemetry.pose
    home = world.home_pose
    near_home = abs(pose.x - home.x) <= 1.0 and abs(pose.y - home.y) <= 1.0
    world.checklist_status.returned_home = near_home
    world.checklist_status.landed = world.telemetry.landed
