"""Construction domain placeholder."""

from __future__ import annotations

from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.domains.base import DomainScenarioBuilder


class ConstructionScenarioBuilder(DomainScenarioBuilder):
    """Not yet implemented; raises so it can't silently masquerade as solar."""

    domain = "construction"

    def build(self, seed: int, episode_id: str | None = None, scenario_family: str | None = None) -> EpisodeWorld:
        raise NotImplementedError(
            "ConstructionScenarioBuilder is not implemented. The MVP focuses on solar; "
            "implement construction-specific assets, viewpoints, and defects before using it."
        )
