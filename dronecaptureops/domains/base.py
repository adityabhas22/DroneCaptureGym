"""Domain interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dronecaptureops.core.state import EpisodeWorld


class DomainScenarioBuilder(ABC):
    """Builds deterministic inspection scenarios."""

    domain: str

    @abstractmethod
    def build(self, seed: int, episode_id: str | None = None) -> EpisodeWorld:
        """Create an episode world for a seed."""
