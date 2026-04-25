"""Centralized scenario suites for solar inspection work."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SuiteEpisode:
    """One deterministic scenario episode."""

    scenario_family: str
    seed: int
    split: str = "train"
    tags: tuple[str, ...] = ()
    max_steps: int | None = None

    @property
    def episode_id(self) -> str:
        return f"{self.scenario_family}:{self.seed}"


@dataclass(frozen=True)
class ScenarioSuite:
    """Named collection of deterministic scenario episodes."""

    name: str
    purpose: str
    episodes: tuple[SuiteEpisode, ...]
    heldout: bool = False

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(episode.scenario_family for episode in self.episodes))

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(dict.fromkeys(episode.seed for episode in self.episodes))


SOLAR_SCENARIO_FAMILIES = (
    "single_hotspot",
    "soiling_and_shadow",
    "bypass_diode_fault",
    "false_positive_glare",
    "blocked_corridor_replan",
    "low_battery_tradeoff",
)


def get_suite(name: str) -> ScenarioSuite:
    """Return a named suite."""

    try:
        return SUITES[name]
    except KeyError as exc:
        available = ", ".join(sorted(SUITES))
        raise ValueError(f"unknown suite: {name}. available suites: {available}") from exc


def list_suites() -> tuple[ScenarioSuite, ...]:
    """Return all scenario suites."""

    return tuple(SUITES[name] for name in sorted(SUITES))


def make_episodes(
    *,
    families: tuple[str, ...],
    seeds: tuple[int, ...],
    split: str = "train",
    tags: tuple[str, ...] = (),
    max_steps: int | None = None,
) -> tuple[SuiteEpisode, ...]:
    """Create all family/seed episode pairs."""

    _validate_families(families)
    return tuple(
        SuiteEpisode(
            scenario_family=family,
            seed=seed,
            split=split,
            tags=tags,
            max_steps=max_steps,
        )
        for family in families
        for seed in seeds
    )


def resolve_suite_episodes(
    *,
    suite_name: str | None = None,
    families: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    max_steps: int | None = None,
) -> tuple[SuiteEpisode, ...]:
    """Resolve a suite or ad-hoc family/seed list into episodes."""

    if suite_name:
        episodes = get_suite(suite_name).episodes
        if max_steps is None:
            return episodes
        return tuple(_with_max_steps(episode, max_steps) for episode in episodes)
    return make_episodes(
        families=families or SOLAR_SCENARIO_FAMILIES,
        seeds=seeds or (2101,),
        split="ad_hoc",
        tags=("ad_hoc",),
        max_steps=max_steps,
    )


def _with_max_steps(episode: SuiteEpisode, max_steps: int) -> SuiteEpisode:
    return SuiteEpisode(
        scenario_family=episode.scenario_family,
        seed=episode.seed,
        split=episode.split,
        tags=episode.tags,
        max_steps=max_steps,
    )


def _validate_families(families: tuple[str, ...]) -> None:
    unknown = sorted(set(families) - set(SOLAR_SCENARIO_FAMILIES))
    if unknown:
        available = ", ".join(SOLAR_SCENARIO_FAMILIES)
        raise ValueError(f"unknown scenario families: {unknown}. available families: {available}")


SUITES: dict[str, ScenarioSuite] = {
    "smoke": ScenarioSuite(
        name="smoke",
        purpose="Fast health check for baseline thermal overview, shadow, and route-safety behavior.",
        episodes=(
            SuiteEpisode("single_hotspot", 2101, tags=("smoke", "easy")),
            SuiteEpisode("soiling_and_shadow", 2102, tags=("smoke", "easy")),
            SuiteEpisode("false_positive_glare", 2203, tags=("smoke", "medium")),
        ),
    ),
    "curriculum_easy": ScenarioSuite(
        name="curriculum_easy",
        purpose="Simple visible rows with one primary anomaly and favorable conditions.",
        episodes=make_episodes(
            families=("single_hotspot", "soiling_and_shadow", "bypass_diode_fault"),
            seeds=(2101, 2102, 2103),
            tags=("curriculum", "easy"),
        ),
    ),
    "curriculum_medium": ScenarioSuite(
        name="curriculum_medium",
        purpose="False-positive cues, partial occlusion, and modality choice pressure.",
        episodes=make_episodes(
            families=("false_positive_glare", "soiling_and_shadow", "bypass_diode_fault"),
            seeds=(2201, 2202, 2203),
            tags=("curriculum", "medium"),
        ),
    ),
    "hard_eval": ScenarioSuite(
        name="hard_eval",
        purpose="Heldout-style stress cases: blocked corridors, low reserve, high wind, and misleading thermal cues.",
        episodes=make_episodes(
            families=("blocked_corridor_replan", "low_battery_tradeoff", "false_positive_glare"),
            seeds=(2301, 2302, 2303),
            split="eval",
            tags=("eval", "hard"),
        ),
    ),
    "demo": ScenarioSuite(
        name="demo",
        purpose="Small narrative set for before/after trajectories.",
        episodes=(
            SuiteEpisode("single_hotspot", 2101, split="demo", tags=("demo", "easy")),
            SuiteEpisode("false_positive_glare", 2203, split="demo", tags=("demo", "medium")),
            SuiteEpisode("blocked_corridor_replan", 2301, split="demo", tags=("demo", "hard")),
        ),
    ),
}
