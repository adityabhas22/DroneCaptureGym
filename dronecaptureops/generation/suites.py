"""Centralized scenario suites for solar inspection work."""

from __future__ import annotations

from dataclasses import dataclass

from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


@dataclass(frozen=True)
class SuiteEpisode:
    """One deterministic scenario episode.

    Two reset modes are supported (mirroring `env.reset` / `ScenarioGenerator.build`):

    * Family-only: `task_id` is empty and the legacy randomized path is used.
    * Task-conditioned: `task_id` is set and the deterministic mission spec
      from `SOLAR_TASKS` overlays the base family scaffold (assets, modalities,
      home/standoff bands). `scenario_family` is still required so the base
      asset `required_modalities` and visibility tags are deterministic.
    """

    scenario_family: str
    seed: int
    split: str = "train"
    tags: tuple[str, ...] = ()
    max_steps: int | None = None
    task_id: str = ""

    @property
    def episode_id(self) -> str:
        if self.task_id:
            return f"task:{self.task_id}:{self.seed}"
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
        task_id=episode.task_id,
    )


def _validate_families(families: tuple[str, ...]) -> None:
    unknown = sorted(set(families) - set(SOLAR_SCENARIO_FAMILIES))
    if unknown:
        available = ", ".join(SOLAR_SCENARIO_FAMILIES)
        raise ValueError(f"unknown scenario families: {unknown}. available families: {available}")


def _validate_task_ids(task_ids: tuple[str, ...]) -> None:
    unknown = sorted(set(task_ids) - set(SOLAR_TASKS))
    if unknown:
        available = ", ".join(sorted(SOLAR_TASKS))
        raise ValueError(f"unknown solar task ids: {unknown}. available task ids: {available}")


def make_task_episodes(
    *,
    task_specs: tuple[tuple[str, str, int], ...],
    split: str = "train",
    tags: tuple[str, ...] = (),
    max_steps: int | None = None,
) -> tuple[SuiteEpisode, ...]:
    """Create deterministic task-conditioned episodes.

    Each entry is `(task_id, base_scenario_family, seed)`. The base family
    seeds the asset modalities and default airspace; the task spec overlays
    mission text, hidden defects, weather, battery, and any extra zones or
    viewpoints. Seeds are kept deterministic per episode for reproducibility.
    """

    families = tuple(family for _, family, _ in task_specs)
    task_ids = tuple(task_id for task_id, _, _ in task_specs)
    _validate_families(families)
    _validate_task_ids(task_ids)
    return tuple(
        SuiteEpisode(
            scenario_family=family,
            seed=seed,
            split=split,
            tags=tags,
            max_steps=max_steps,
            task_id=task_id,
        )
        for task_id, family, seed in task_specs
    )


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
    "demo_llm_inspection": ScenarioSuite(
        name="demo_llm_inspection",
        purpose=(
            "Small task-conditioned suite for the live rich-sim UI: basic movement and capture, "
            "anomaly confirmation, obstacle detours, low-battery return pressure, and privacy-safe evidence."
        ),
        episodes=make_task_episodes(
            task_specs=(
                ("basic_thermal_survey", "single_hotspot", 2501),
                ("anomaly_confirmation", "single_hotspot", 2502),
                ("obstacle_detour_inspection", "blocked_corridor_replan", 2503),
                ("low_battery_inspection", "low_battery_tradeoff", 2504),
                ("privacy_safe_alternate_evidence", "false_positive_glare", 2505),
            ),
            split="demo",
            tags=("demo", "live", "rich_sim"),
            max_steps=40,
        ),
    ),
    "solar_tasks": ScenarioSuite(
        name="solar_tasks",
        purpose=(
            "Task-conditioned solar benchmark: 45 mechanically distinct inspection-director missions "
            "covering coverage baselines, anomaly confirmation, recapture loops, zoom, scope control, "
            "privacy, routing, dynamic obstacles, return-margin decisions, weighted triage, false "
            "positives, and grounded reporting."
        ),
        episodes=make_task_episodes(
            task_specs=(
                ("basic_thermal_survey", "single_hotspot", 2401),
                ("anomaly_confirmation", "single_hotspot", 2402),
                ("low_battery_inspection", "low_battery_tradeoff", 2403),
                ("inspect_recapture_quality_loop", "single_hotspot", 2404),
                ("compound_safety_corridor", "blocked_corridor_replan", 2405),
                ("honest_partial_report_open_items", "low_battery_tradeoff", 2406),
                ("multi_anomaly_triage", "bypass_diode_fault", 2407),
                ("zoom_required_long_standoff", "bypass_diode_fault", 2408),
                ("edge_row_quality_bar", "single_hotspot", 2409),
                ("no_anomaly_clearance", "false_positive_glare", 2410),
                ("obstacle_detour_inspection", "blocked_corridor_replan", 2411),
                ("privacy_zone_capture", "false_positive_glare", 2412),
                ("soft_privacy_capture_positioning", "false_positive_glare", 2413),
                ("thermal_only_anomaly_skip_rgb", "single_hotspot", 2414),
                ("multi_anomaly_routing_under_obstacle", "blocked_corridor_replan", 2415),
                ("single_row_reinspection", "single_hotspot", 2416),
                ("pid_multi_row_pattern", "bypass_diode_fault", 2417),
                ("diode_fault_needs_close_thermal", "bypass_diode_fault", 2418),
                ("bird_soiling_explanation", "soiling_and_shadow", 2419),
                ("vegetation_edge_encroachment", "soiling_and_shadow", 2420),
                ("substation_adjacency_caution", "single_hotspot", 2421),
                ("strict_severity_weighted_triage", "low_battery_tradeoff", 2422),
                ("true_false_anomaly_discrimination", "false_positive_glare", 2423),
                ("permanent_occlusion_coverage", "blocked_corridor_replan", 2424),
                ("prioritized_triage_under_constraint", "low_battery_tradeoff", 2425),
                ("capture_efficiency_discipline", "single_hotspot", 2426),
                ("partial_blocked_anomaly_honest_report", "false_positive_glare", 2427),
                ("no_defect_with_glare_artifact", "false_positive_glare", 2428),
                ("required_rows_subset_priority", "single_hotspot", 2429),
                ("audit_grade_strict_grounding", "bypass_diode_fault", 2430),
                ("return_margin_decision_point", "low_battery_tradeoff", 2431),
                ("route_replan_when_primary_viewpoint_blocked", "blocked_corridor_replan", 2432),
                ("scheduled_crane_window_wait_or_detour", "blocked_corridor_replan", 2433),
                ("minimum_evidence_for_dispatch", "low_battery_tradeoff", 2434),
                ("post_repair_verification", "single_hotspot", 2435),
                ("warranty_claim_evidence_pack", "bypass_diode_fault", 2436),
                ("operator_abort_under_safety_pressure", "low_battery_tradeoff", 2437),
                ("privacy_safe_alternate_evidence", "false_positive_glare", 2438),
                ("glare_angle_experiment", "false_positive_glare", 2439),
                ("quality_vs_efficiency_tradeoff", "single_hotspot", 2440),
                ("multi_issue_one_rgb_context", "bypass_diode_fault", 2441),
                ("thermal_only_fast_clearance", "single_hotspot", 2442),
                ("low_severity_ignore_under_budget", "low_battery_tradeoff", 2443),
                ("blocked_return_path_requires_safe_dogleg", "blocked_corridor_replan", 2444),
                ("commissioning_acceptance_survey", "single_hotspot", 2445),
            ),
            tags=("task", "benchmark"),
        ),
    ),
}
