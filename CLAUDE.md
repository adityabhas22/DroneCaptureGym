# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Install (Python 3.11+):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"     # add `,train` for TRL/accelerate/peft/datasets
```

Test:

```bash
pytest                                  # full suite (deterministic, no SITL)
pytest tests/test_rewards.py            # one file
pytest tests/test_rewards.py::test_name # one test
```

Run an episode locally:

```bash
python examples/run_scripted_agent.py
python examples/run_random_agent.py
```

Run scenario suites and traces:

```bash
python -m training.run_suite --suite smoke --policy scripted
python -m training.run_suite --suite hard_eval --policy weak_scripted --output report.json
python -m training.trace_episode --suite demo --episode-index 0 --policy scripted --output-dir artifacts/trace-demo
# Console scripts (after `pip install -e .`):
dronecaptureops-run-suite --suite smoke
dronecaptureops-trace-episode --suite demo --episode-index 0
```

Available suites: `smoke`, `curriculum_easy`, `curriculum_medium`, `hard_eval`, `demo` (see `dronecaptureops/generation/suites.py`). Available policies: `random`, `weak_scripted`, `scripted`.

OpenEnv HTTP server:

```bash
dronecaptureops-server                          # console script
uvicorn server.app:app --host 0.0.0.0 --port 8000   # equivalent
```

`openenv.yaml` points to `server.app:app`. `DRONECAPTUREOPS_MAX_CONCURRENT_ENVS` (default 64) caps concurrent sessions.

## Architecture

DroneCaptureOps Gym is an OpenEnv-compatible RL environment where the agent acts as an **inspection director**: it never controls motors, only issues high-level tool calls (mission review, waypoint flight, gimbal/camera control, RGB/thermal capture, capture inspection, report submission). The benchmark is active visual inspection — collecting grounded evidence — not navigation.

### Layered packages and their contracts

```
core/           OpenEnv env, typed Pydantic models, action validation, constants, errors
controllers/    DroneController abstraction → GeometryController (default), DroneKitSITL placeholder
simulation/     geometry, camera quality, battery, weather, safety checks, world updates
domains/        DomainScenarioBuilder per domain (solar is the only fleshed-out one)
generation/     ScenarioGenerator dispatch, ScenarioSuite definitions, seed normalization
tools/          public ToolRegistry + flight/camera/inspection/report tool handlers
rewards/        composable reward components + RewardAggregator + verifier utilities
evaluation/     RolloutRunner, baseline policies, suite_runner, tracing artifacts
utils/          math, geo, logging, serialization helpers
server/         FastAPI app for OpenEnv deployment
training/       eval + run_suite + trace_episode + GRPO scaffold
```

The `DroneCaptureOpsEnvironment` (`core/environment.py`) wires it all together: on `reset` it asks `ScenarioGenerator` for an `EpisodeWorld`, resets the controller, and returns a `DroneObservation`. On `step` it coerces the action into `RawDroneAction`, runs `ToolRegistry.validate` then `.execute`, lets `SafetyChecker` reject unsafe calls, updates terminal status, computes a `RewardBreakdown` via `RewardAggregator`, and renders the next observation including `inspection_affordances` (mission phase, blockers, suggested tools).

### Hidden vs. visible state — load-bearing invariant

`EpisodeWorld` (`core/state.py`) holds verifier-only fields: `hidden_defects`, `true_asset_state`, `hidden_weather_details`, `obstacle_schedule`, `verifier_evidence_requirements`. Observations and `DroneVisibleState` MUST NOT leak these. Final reports must cite real artifact IDs that were actually captured; the integrity gate caps reward when a report references fake photo IDs or unsupported claims. `tests/test_no_hidden_state_leakage.py` enforces this — preserve it when adding new observation fields.

### Reward shape

`RewardAggregator.compute` produces a `RewardBreakdown` and stores it on the world. The total is:

```
total = clamp(min(safety_gate, integrity_gate,
                  0.45*evidence_success + 0.20*required_coverage + 0.15*issue_capture
                + 0.10*operational_efficiency + 0.10*grounded_report
                + process_reward - penalties),
              -1, 1)
```

`safety_gate` and `integrity_gate` are caps, not multipliers — a no-fly violation caps total at ~0.10; fake photo IDs cap via integrity. Old fields (`target_coverage`, `defect_visibility`, `route_efficiency`, `report_grounding`) are mirrored from the new ones for backward compatibility — keep populating both. `process_reward` is bounded at 0.10 and exists to provide a dense learning signal; do not let it dominate outcome rewards. Verifier checks live in `rewards/verifiers.py` and use simulator state + capture metadata only (no LLM judge).

### Tool surface

`build_tool_registry` (`tools/__init__.py`) is the single source of truth for the public action schema. Each `ToolSpec` declares `required`/`optional` arg sets that drive validation in `ToolRegistry.validate`. `ToolRegistry.action_availability` derives the visible availability map from public state (in_air, landed, has_capture, returned_home, landed). When adding a tool: register it here, implement the handler on the matching `*Tools` class, and update `_suggested_tools` / `_recommended_action_categories` in `core/environment.py` if it changes mission phase guidance.

### Adding new domains, controllers, or rewards

- **New domain:** subclass `DomainScenarioBuilder` (`domains/base.py`), set `domain` attribute, return a fully populated `EpisodeWorld`, and register the builder in `ScenarioGenerator.__init__`. Solar is the reference implementation; bridge/construction/industrial are stubs.
- **New scenario family:** add to `SOLAR_SCENARIO_FAMILIES` in `generation/suites.py` and have the domain builder branch on `scenario_family`.
- **New controller backend:** implement `DroneController` (`controllers/base.py`). The default `GeometryController` is deterministic and CPU-light. `DroneKitSITLController` is a placeholder — keep SITL/ArduPilot work behind this interface.
- **New reward component:** add a class under `rewards/`, instantiate in `RewardAggregator.__init__`, fold into `compute()`, and add a field to `RewardBreakdown` in `core/models.py`. Add coverage in `tests/test_rewards.py`.

### Determinism

Every episode is reproducible from `(domain, scenario_seed, scenario_family)`. Use `normalize_seed` (`generation/seeds.py`) when accepting seeds. Tests rely on this — `RolloutRunner` and `RandomPolicy` both take explicit seeds.

## Engineering principles (from CODEX.md)

- **Non-goals:** no raw motor control; no exposing hidden state; no shortest-path-only rewards; no easily-gameable rewards; no coupling to a single simulator backend; no domain-specific logic in generic core classes.
- Keep the MVP geometry-first. Do not add photorealistic rendering or full DroneKit integration until the fast simulator, reward shape, and evidence verification are stable.
- Centralize constants in `core/constants.py`, action schemas in `tools/__init__.py`, and quality/safety thresholds in `simulation/`.
- Important logic must be testable without launching a heavy simulator.
