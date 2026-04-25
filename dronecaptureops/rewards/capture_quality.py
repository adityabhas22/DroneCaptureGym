"""Capture-quality reward."""

from __future__ import annotations

from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.base import RewardComponent


class CaptureQualityReward(RewardComponent):
    """Rewards high-quality non-empty captures."""

    name = "capture_quality"

    def compute(self, world: EpisodeWorld) -> float:
        useful = [capture.quality_score for capture in world.capture_log if capture.targets_visible]
        if not useful:
            return 0.0
        return round(sum(useful) / len(useful), 4)
