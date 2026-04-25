# DroneCaptureOps Gym

DroneCaptureOps Gym is an OpenEnv-compatible environment for training LLM agents to operate as aerial inspection directors. The agent does not control motors. It acts through high-level tools for mission review, safe waypoint flight, gimbal/camera control, RGB and thermal capture, capture inspection, return-home, and final evidence-pack submission.

The benchmark focus is active visual inspection: deciding what evidence is missing, where to capture next, when to recapture, and whether the final evidence pack is grounded in real photos.

## What Exists Now

- Deterministic `FastGeometrySim` style backend through `GeometryController`.
- Solar inspection MVP with rows `B4-B8`, a substation no-fly zone, seeded hidden defects, weather, and battery use.
- OpenEnv `reset`, `step`, and visible `state` support.
- High-level tool registry with schema validation.
- Safety wrapper before flight/gimbal actions.
- Structured RGB/thermal capture-quality metadata.
- DroneKit-inspired rich telemetry and generic inspection assets documented in [docs/environment-model-v2.md](docs/environment-model-v2.md).
- Solar scenario families and tool-surface research notes in [docs/solar-scenario-research.md](docs/solar-scenario-research.md).
- Composable reward breakdown with safety gate and report-grounding checks.
- Fast pytest coverage for reset, step, safety, rewards, scenario generation, and hidden-state protection.
- Placeholder `DroneKitSITLController` adapter boundary for future ArduPilot / DroneKit work.

## Project Structure

```text
dronecaptureops/
  core/          environment loop, models, state, actions, observations
  controllers/   DroneController abstraction, GeometryController, SITL placeholder
  simulation/    geometry, camera, battery, weather, safety
  domains/       scenario builders, starting with solar
  rewards/       reward components and aggregation
  tools/         public tool registry and handlers
  generation/    domain/seed scenario generator
  utils/         math, geo, logging, serialization helpers
server/          OpenEnv FastAPI entrypoint
training/        GRPO/eval scaffolds
examples/        random and scripted rollouts
tests/           fast deterministic tests
```

## Install

Requires Python 3.11 or newer.

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"        # add ",train" for TRL/accelerate/peft/datasets
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"        # add ",train" for TRL/accelerate/peft/datasets
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

## First Five Commands

A new contributor can validate the environment end-to-end with:

```bash
pytest                                                 # 1. fast deterministic test suite
python examples/run_scripted_agent.py                  # 2. run one solved episode locally
python -m training.run_suite --suite smoke --policy scripted   # 3. run a baseline suite
python -m training.trace_episode --suite demo --episode-index 0 --policy scripted --output-dir artifacts/trace-demo   # 4. dump a per-step trace
dronecaptureops-server                                 # 5. start the OpenEnv HTTP server (Ctrl-C to stop)
```

## Run Tests

```bash
pytest
pytest tests/test_rewards.py                            # one file
pytest tests/test_rewards.py::test_reward_breakdown_shape   # one test
```

## Run One Episode

```bash
python examples/run_scripted_agent.py
python examples/run_random_agent.py
python examples/run_task_suite.py        # scripted rollout across every task in SOLAR_TASKS
```

## Run Suites and Trace Trajectories

```bash
python -m training.run_suite --suite smoke --policy scripted
python -m training.run_suite --suite hard_eval --policy weak_scripted --output report.json
python -m training.trace_episode --suite demo --episode-index 0 --policy scripted --output-dir artifacts/trace-demo
```

Available suites: `smoke`, `curriculum_easy`, `curriculum_medium`, `hard_eval`, `demo` (defined in `dronecaptureops/generation/suites.py`). Available policies: `random`, `weak_scripted`, `scripted`.

Trace output includes `episode_steps.json`, `evidence_log.json`, `route_log.json`, `inspection_report.json`, `trace.json`, and `trace.md`. Each step records the action, next observation, reward breakdown, reward deltas, warnings, and visible state changes so training runs can be debugged trajectory by trajectory.

## Scenario Suites vs. Task IDs

There are two orthogonal selectors and they do different things:

- **Task ID** (`env.reset(task="anomaly_confirmation")`, alias `task_id=`) picks a deterministic *mission variant* from `dronecaptureops/tasks/solar_tasks.py`. The 15 solar tasks override hidden defects, weather, battery, step budget, extra zones/viewpoints, and quality thresholds. They are the unit of RL task-conditioning.
- **Scenario suite** (`--suite smoke`) is a named bundle of `(scenario_family, seed)` pairs from `dronecaptureops/generation/suites.py` used for evaluation runs. Each suite episode uses the legacy seed/family randomized path, **not** a task ID. Suites are the unit of benchmark/regression reporting.

Both go through the same `DroneCaptureOpsEnvironment.reset()` and produce a fully-formed `EpisodeWorld`; they just route through different branches of `SolarScenarioBuilder.build`. A task ID overrides scenario-family defaults when both are passed.

## OpenEnv Server

```bash
dronecaptureops-server
```

or:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

`openenv.yaml` points to `server.app:app`.

## Action Shape

Actions are structured tool calls:

```json
{
  "tool_name": "fly_to_viewpoint",
  "arguments": {
    "x": 30,
    "y": 0,
    "z": 18,
    "yaw_deg": 180,
    "speed_mps": 5
  }
}
```

Initial tools:

- Mission/map: `get_site_map`, `get_mission_checklist`, `get_telemetry`, `list_assets`, `estimate_view`, `estimate_return_margin`, `request_route_replan`
- Flight: `takeoff`, `fly_to_viewpoint`, `move_to_asset`, `hover`, `return_home`, `land`
- Camera: `set_gimbal`, `set_zoom`, `set_camera_source`, `point_camera_at`, `capture_rgb`, `capture_thermal`, `inspect_capture`
- Evidence: `mark_target_inspected`, `submit_evidence_pack`

## Reward Shape

The terminal reward formula (after `submit_evidence_pack`) is:

```
total = clamp(min(safety_gate, integrity_gate,
                  0.45*evidence_success
                + 0.20*required_coverage
                + 0.15*issue_capture
                + 0.10*operational_efficiency
                + 0.10*grounded_report
                + process_reward
                - penalties),
              -1, 1)
```

`safety_gate` and `integrity_gate` are caps, not multipliers — a no-fly violation caps total at ~0.10; fake or unsupported photo citations cap via integrity. `process_reward` is bounded at 0.10. The legacy fields `target_coverage`, `defect_visibility`, `route_efficiency`, and `report_grounding` are mirrored from the new ones for backward compatibility.

### Terminal vs. Non-Terminal Behavior

The reward is published every step but its meaning changes when the agent calls `submit_evidence_pack`:

- **Before submission** — `total` is a small *shaping* reward derived from captured (not cited) progress, used as a dense learning signal. Outcome-grade scoring is deliberately suppressed so the agent cannot farm reward by hovering and capturing forever.
- **After submission** — `total` is the actual mission outcome score using *cited* evidence, with safety and integrity caps applied. `done=True`.

The `reward_breakdown.debug` payload makes this explicit so training code can branch correctly:

```json
// non-terminal step
{
  "terminal_submitted": false,
  "nonterminal_cap_applied": true,
  "shaping_reward": 0.14,
  "raw_outcome_if_submitted": 0.72
}

// terminal step (after submit_evidence_pack)
{
  "terminal_submitted": true,
  "nonterminal_cap_applied": false,
  "shaping_reward": 0.14,
  "raw_outcome_if_submitted": 0.91
}
```

Components logged on every step include: `format_validity`, `flight_success`, `evidence_success`, `required_coverage`, `issue_capture`, `operational_efficiency`, `grounded_report`, `process_reward`, `integrity_gate`, `value_per_photo`, `capture_quality`, `checklist_completion`, `battery_management`, `safety_compliance`, `recovery_behavior`, `penalties`, `safety_gate`, `total`, plus the legacy mirrors above.

## Hidden State

Hidden defects and verifier labels live only in internal `EpisodeWorld`. Observations and visible state expose mission, map, telemetry, captures, reward breakdown, and checklist status, but not the hidden expected solution.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add a new tool, reward component, task, or domain. Each section lists the files to touch, contracts to honor, and tests to add.

## Demo Direction

A weak baseline should fly to a generic overview point, capture one or two low-value photos, miss an anomaly or close-up, and submit an incomplete pack. A stronger trained agent should read the checklist, capture thermal overview, inspect quality, reposition for missing rows or anomalies, capture RGB close-up evidence, avoid the no-fly zone, return home, land, and submit a photo-linked evidence pack.
