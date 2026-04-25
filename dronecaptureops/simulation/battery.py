"""Battery model for the fast geometry simulator."""

from __future__ import annotations

from dronecaptureops.core.constants import (
    CAPTURE_BATTERY_COST,
    FLIGHT_BATTERY_COST_PER_M,
    HOVER_BATTERY_COST_PER_S,
    LAND_BATTERY_COST,
    TAKEOFF_BATTERY_COST,
)
from dronecaptureops.utils.math_utils import clamp


def drain_for_takeoff(battery_pct: float) -> float:
    """Apply takeoff battery drain."""

    return clamp(battery_pct - TAKEOFF_BATTERY_COST, 0.0, 100.0)


def drain_for_flight(battery_pct: float, distance_m: float) -> float:
    """Apply flight battery drain."""

    return clamp(battery_pct - distance_m * FLIGHT_BATTERY_COST_PER_M, 0.0, 100.0)


def drain_for_hover(battery_pct: float, seconds: float) -> float:
    """Apply hover battery drain."""

    return clamp(battery_pct - seconds * HOVER_BATTERY_COST_PER_S, 0.0, 100.0)


def drain_for_capture(battery_pct: float) -> float:
    """Apply image capture battery drain."""

    return clamp(battery_pct - CAPTURE_BATTERY_COST, 0.0, 100.0)


def drain_for_land(battery_pct: float) -> float:
    """Apply landing battery drain."""

    return clamp(battery_pct - LAND_BATTERY_COST, 0.0, 100.0)
