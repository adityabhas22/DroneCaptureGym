"""Battery-management reward component.

Route efficiency lives in `verifiers.compute_operational_efficiency`;
this module only owns the battery-reserve component now.
"""

from __future__ import annotations

from dronecaptureops.core.constants import MIN_BATTERY_TO_RETURN_PCT
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.base import RewardComponent
from dronecaptureops.utils.math_utils import clamp


class BatteryManagementReward(RewardComponent):
    """Rewards returning or finishing with reserve battery."""

    name = "battery_management"

    def compute(self, world: EpisodeWorld) -> float:
        reserve = world.telemetry.battery.level_pct - MIN_BATTERY_TO_RETURN_PCT
        return round(clamp(reserve / 55.0, 0.0, 1.0), 4)
