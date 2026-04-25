"""Safety reward."""

from __future__ import annotations

from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.base import RewardComponent


class SafetyReward(RewardComponent):
    """Rewards avoiding safety violations."""

    name = "safety_compliance"

    def compute(self, world: EpisodeWorld) -> float:
        if not world.safety_violations:
            return 1.0
        return max(0.0, round(1.0 - 0.35 * len(world.safety_violations), 4))
