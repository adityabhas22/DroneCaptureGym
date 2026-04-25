"""Reward component interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dronecaptureops.core.state import EpisodeWorld


class RewardComponent(ABC):
    """Computes one reward component from internal world state."""

    name: str

    @abstractmethod
    def compute(self, world: EpisodeWorld) -> float:
        """Return a normalized reward value."""
