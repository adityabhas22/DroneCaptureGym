# Contributing

This guide is for adding the four most common kinds of extension to DroneCaptureOps Gym: a tool, a reward component, a task, or a domain. Each section lists the files to touch, the contracts to honor, and the tests to add. Before extending anything, run `pytest` to make sure the baseline is green.

The high-level invariants from `CLAUDE.md` apply to every change:

- The agent never sees hidden state. Anything stored on `EpisodeWorld` that is not on `DroneObservation`/`DroneVisibleState` is verifier-only.
- Episodes are reproducible from `(domain, scenario_seed, scenario_family, task_id)`. New code that introduces randomness must take a seed.
- Important logic must be testable without launching SITL or rendering imagery.

## Adding a New Tool

The public action surface is defined entirely by `build_tool_registry` in `dronecaptureops/tools/__init__.py`. A tool is two things together: a `ToolSpec` (validation contract) and a handler method on one of the `*Tools` classes.

1. Implement the handler on the matching class:
   - flight verbs → `dronecaptureops/tools/flight_tools.py`
   - camera/gimbal/capture → `dronecaptureops/tools/camera_tools.py`
   - read-only inspection / map / telemetry → `dronecaptureops/tools/inspection_tools.py`
   - final report → `dronecaptureops/tools/report_tools.py`

   The signature is `handler(world: EpisodeWorld, action: RawDroneAction) -> dict[str, Any]`. Returning a `{"error": "..."}` dict marks the call as failed without raising. Raise `SafetyViolationError` from `core/errors` for unsafe calls so the env routes the failure through the safety gate.

2. Register it in `build_tool_registry`:

   ```python
   registry.register(ToolSpec(
       "estimate_wind_correction",
       "Return suggested heading offset for current wind.",
       required={"target_yaw_deg"},
       optional={"altitude_m"},
       handler=inspection.estimate_wind_correction,
   ))
   ```

   `ToolRegistry.validate` enforces required/optional arg sets, so callers get clear errors before the handler runs.

3. If the new tool changes mission flow, update `_suggested_tools` and `_recommended_action_categories` in `dronecaptureops/core/environment.py`. If it gates on a new piece of public state, also extend `ToolRegistry.action_availability` so `inspection_affordances.action_availability` stays accurate.

4. Add tests in `tests/test_tool_calling_surface.py` (schema/availability) and `tests/test_environment_step.py` (end-to-end behavior). Include at least one happy path and one validation failure (missing arg, wrong type, or unsafe range).

## Adding a New Reward Component

Rewards compose in `dronecaptureops/rewards/reward_aggregator.py` and are logged on `RewardBreakdown` in `dronecaptureops/core/models.py`.

1. Add a small class under `dronecaptureops/rewards/` with a `compute(world: EpisodeWorld) -> float` method (see `safety.py` or `efficiency.py` for the pattern). Keep it deterministic and pure — no LLM judges, no I/O. Verifier helpers belong in `dronecaptureops/rewards/verifiers.py`.

2. Add a field to `RewardBreakdown` in `dronecaptureops/core/models.py` with a sensible default (0.0 for additive components, 1.0 for caps).

3. Instantiate it in `RewardAggregator.__init__` and fold its value into the breakdown built inside `RewardAggregator.compute`. Decide whether the new component:
   - is an additive shaped term (multiply by a small weight inside `raw_outcome` and `raw_outcome_if_submitted`),
   - is a *cap* like `safety_gate`/`integrity_gate` (apply via `min(...)` inside the terminal branch only),
   - or is a *penalty* (subtract through `_penalties`).

   `process_reward` is intentionally bounded at 0.10. Do not bypass that bound — outcome reward must dominate.

4. Mirror the value into any legacy field if the new term replaces an old one (see how `target_coverage` mirrors `required_coverage`). Keep both populated until consumers migrate.

5. Add coverage in `tests/test_rewards.py`. Cover at minimum: the no-op case, the positive case, the gate/penalty interaction with `submit_evidence_pack`, and farming resistance for non-terminal steps.

## Adding a New Task

Tasks are deterministic mission variants defined in `dronecaptureops/tasks/solar_tasks.py`. They share the same `EpisodeWorld` shape but override defects, weather, battery, step budget, extra zones/viewpoints, and quality thresholds.

1. Add a `SolarTaskSpec` entry to the `SOLAR_TASKS` dict. Use `hidden_defects=()` for "no anomaly is present" missions and `hidden_defects=None` only when you explicitly want to inherit nothing (in practice both are equivalent for the task path — see the docstring on `_hidden_defects_from_task`). Use deterministic `DefectSpec` IDs that downstream tests can assert against.

2. Set quality thresholds (`min_capture_quality`, `min_rgb_quality`, `min_report_grounding_score`) to express the mission's strictness. The verifier reads these directly.

3. If the task adds airspace constraints or capture viewpoints, populate `extra_zones` and `extra_viewpoints`. These flow through `SolarScenarioBuilder.build` into the visible site map.

4. If the task needs new env-level constraints (e.g. a privacy zone capture rule, return-home compliance), confirm the existing safety/verifier code in `dronecaptureops/simulation/safety.py` and `dronecaptureops/rewards/verifiers.py` already handles it. Otherwise extend those modules first and treat that as a separate change.

5. Add a test in `tests/test_solar_tasks.py` that resets the env with the new `task_id`, walks through enough steps to satisfy success criteria, and asserts the relevant reward components and hidden-state invariants.

## Adding a New Domain

Domains are scenario builders that produce a fully populated `EpisodeWorld` for a different inspection setting (bridge, construction, industrial, etc.). Solar is the only fleshed-out reference.

1. Subclass `DomainScenarioBuilder` from `dronecaptureops/domains/base.py`. Set the class-level `domain` attribute to the canonical string used in `env.reset(domain=...)`.

2. Implement `build(seed, episode_id, scenario_family, task_id) -> EpisodeWorld`. The returned world must populate:
   - assets, viewpoints, airspace zones, home pose, telemetry, weather,
   - `mission` (a `MissionChecklist` with required modalities/rows and constraints),
   - `hidden_defects`, `true_asset_state`, `hidden_weather_details`, `obstacle_schedule`, `verifier_evidence_requirements` — all of which stay verifier-only.

   Use `random.Random(seed)` for any sampling so the same seed always produces the same world.

3. Register the builder in `ScenarioGenerator.__init__` (`dronecaptureops/generation/scenario_generator.py`).

4. If the domain has its own scenario families, add them to a domain-specific tuple (mirror `SOLAR_SCENARIO_FAMILIES`) and have the builder branch on `scenario_family`. If the domain is task-conditioned, mirror the `tasks/solar_tasks.py` pattern under `dronecaptureops/tasks/`.

5. Add a smoke test in `tests/test_scenario_generation.py` that resets with the new domain and asserts the world is well-formed and free of hidden-state leakage. The leakage test in `tests/test_no_hidden_state_leakage.py` enforces the cross-cutting invariant — make sure it passes for the new domain too.

## Local Validation Checklist

Before opening a PR, run:

```bash
pytest                                                              # all tests
python examples/run_scripted_agent.py                               # reward should print 1.0
python -m training.run_suite --suite smoke --policy scripted        # suite report should be green
```

If you touched the OpenEnv surface, also start `dronecaptureops-server` and confirm `/metadata` and `/schema` respond.
