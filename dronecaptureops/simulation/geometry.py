"""Geometry helpers for the fast simulator."""

from __future__ import annotations

from dronecaptureops.core.models import Pose
from dronecaptureops.utils.math_utils import distance_3d


def flight_distance(start: Pose, end: Pose) -> float:
    """Return flight distance between two poses."""

    return distance_3d(start, end)
