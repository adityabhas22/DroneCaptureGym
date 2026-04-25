"""Domain scenario generator registry."""

from __future__ import annotations

from dronecaptureops.core.constants import DEFAULT_DOMAIN
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.domains.base import DomainScenarioBuilder
from dronecaptureops.domains.bridge import BridgeScenarioBuilder
from dronecaptureops.domains.construction import ConstructionScenarioBuilder
from dronecaptureops.domains.industrial import IndustrialScenarioBuilder
from dronecaptureops.domains.solar import SolarScenarioBuilder
from dronecaptureops.generation.seeds import normalize_seed


class ScenarioGenerator:
    """Build deterministic scenarios for supported domains."""

    def __init__(self, builders: dict[str, DomainScenarioBuilder] | None = None) -> None:
        self._builders = builders or {
            "solar": SolarScenarioBuilder(),
            "construction": ConstructionScenarioBuilder(),
            "bridge": BridgeScenarioBuilder(),
            "industrial": IndustrialScenarioBuilder(),
        }

    def build(self, seed: int | None = None, domain: str = DEFAULT_DOMAIN, episode_id: str | None = None) -> EpisodeWorld:
        if domain not in self._builders:
            supported = ", ".join(sorted(self._builders))
            raise ValueError(f"unsupported domain {domain!r}; supported domains: {supported}")
        return self._builders[domain].build(seed=normalize_seed(seed), episode_id=episode_id)
