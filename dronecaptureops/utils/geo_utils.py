"""Map-specific geometry helpers."""

from __future__ import annotations

from dronecaptureops.core.models import Pose, RectZone


def point_in_rect(x: float, y: float, zone: RectZone) -> bool:
    """Return whether a point lies inside a rectangular zone."""

    return zone.min_x <= x <= zone.max_x and zone.min_y <= y <= zone.max_y


def pose_in_zone(pose: Pose, zone: RectZone) -> bool:
    """Return whether a pose lies inside a zone."""

    return point_in_rect(pose.x, pose.y, zone)


def segment_intersects_rect(start: Pose, end: Pose, zone: RectZone, samples: int = 24) -> bool:
    """Approximate whether a segment crosses a rectangular zone."""

    for idx in range(samples + 1):
        t = idx / samples
        x = start.x + (end.x - start.x) * t
        y = start.y + (end.y - start.y) * t
        if point_in_rect(x, y, zone):
            return True
    return False
