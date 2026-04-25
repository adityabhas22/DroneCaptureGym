"""Small geometry helpers."""

from __future__ import annotations

import math

from dronecaptureops.core.models import Pose


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a value into an inclusive range."""

    return max(low, min(high, value))


def distance_2d(a: Pose, x: float, y: float) -> float:
    """Return planar distance from a pose to a point."""

    return math.hypot(a.x - x, a.y - y)


def distance_3d(a: Pose, b: Pose) -> float:
    """Return Euclidean distance between two poses."""

    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def bearing_deg(from_pose: Pose, x: float, y: float) -> float:
    """Return local bearing from pose to point in degrees."""

    return math.degrees(math.atan2(y - from_pose.y, x - from_pose.x))


def angle_delta_deg(a: float, b: float) -> float:
    """Return the smallest absolute angular difference."""

    return abs((a - b + 180.0) % 360.0 - 180.0)
