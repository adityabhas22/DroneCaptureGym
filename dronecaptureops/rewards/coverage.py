"""Target coverage reward."""

from __future__ import annotations

from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.base import RewardComponent


class TargetCoverageReward(RewardComponent):
    """Rewards thermal coverage of required mission rows."""

    name = "target_coverage"

    def compute(self, world: EpisodeWorld) -> float:
        required = set(world.mission.required_rows)
        if not required:
            return 1.0
        covered = set(world.checklist_status.thermal_rows_covered)
        return round(len(required & covered) / len(required), 4)
