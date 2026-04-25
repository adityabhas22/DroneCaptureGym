"""Route efficiency and battery reward components."""

from __future__ import annotations

from dronecaptureops.core.constants import MIN_BATTERY_TO_RETURN_PCT
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.base import RewardComponent
from dronecaptureops.utils.math_utils import clamp


class RouteEfficiencyReward(RewardComponent):
    """Rewards collecting evidence without excessive travel."""

    name = "route_efficiency"

    def compute(self, world: EpisodeWorld) -> float:
        if world.distance_flown_m <= 0:
            return 0.0
        efficient_distance = 95.0
        return round(clamp(1.0 - max(0.0, world.distance_flown_m - efficient_distance) / 150.0, 0.0, 1.0), 4)


class BatteryManagementReward(RewardComponent):
    """Rewards returning or finishing with reserve battery."""

    name = "battery_management"

    def compute(self, world: EpisodeWorld) -> float:
        reserve_floor = max(MIN_BATTERY_TO_RETURN_PCT, world.mission.min_battery_at_done_pct)
        reserve = world.telemetry.battery_pct - reserve_floor
        return round(clamp(reserve / 55.0, 0.0, 1.0), 4)
