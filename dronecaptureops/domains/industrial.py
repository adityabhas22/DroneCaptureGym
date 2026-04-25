"""Industrial domain placeholder."""

from __future__ import annotations

from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.domains.base import DomainScenarioBuilder


class IndustrialScenarioBuilder(DomainScenarioBuilder):
    """Not yet implemented; raises so it can't silently masquerade as solar."""

    domain = "industrial"

    def build(self, seed: int, episode_id: str | None = None, scenario_family: str | None = None) -> EpisodeWorld:
        raise NotImplementedError(
            "IndustrialScenarioBuilder is not implemented. The MVP focuses on solar; "
            "implement industrial-specific assets, viewpoints, and defects before using it."
        )
