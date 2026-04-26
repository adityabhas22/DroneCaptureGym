# DroneCaptureOps Gym

DroneCaptureOps Gym is an OpenEnv-compatible environment for training LLM agents to operate as aerial inspection directors. The agent does not control motors. It acts through high-level tools for mission review, safe waypoint flight, gimbal/camera control, RGB and thermal capture, capture inspection, return-home, and final evidence-pack submission.

The benchmark focus is active visual inspection: deciding what evidence is missing, where to capture next, when to recapture, and whether the final evidence pack is grounded in real photos.

## Hackathon Submission Status

DroneCaptureOps Gym is a high-level aerial inspection environment for OpenEnv-style RL agents. It matters because real inspection agents must gather grounded RGB/thermal evidence, obey safety constraints, and submit reports that cite actual captured artifacts rather than hidden verifier state or invented photo IDs.

Current submission links and evidence:

- Hugging Face Space: `TODO_HF_SPACE_URL`
- Executed training notebook: `TODO_EXECUTED_TRAINING_NOTEBOOK_URL`
- Experiment tracking: `TODO_EXPERIMENT_TRACKING_URL`
- Loss plot: `TODO_LOSS_PLOT_URL`
- Reward plot: `TODO_REWARD_PLOT_URL`
- Trained model or adapter: `TODO_TRAINED_MODEL_OR_ADAPTER_URL`
- Writeup or video: `TODO_WRITEUP_OR_VIDEO_URL`
- Optional presentation: `TODO_OPTIONAL_PRESENTATION_URL`

Real training evidence is pending until an actual training run is completed and the tracker, loss plot, reward plot, and model or adapter links above are replaced with real URLs. Do not treat placeholder links as submitted evidence.

Environment summary:

- Action space: structured `RawDroneAction` tool calls with `tool_name` and `arguments`, covering mission review, map queries, safe flight, gimbal/camera control, RGB/thermal capture, capture inspection, return-home, landing, and evidence-pack submission.
- Observation space: visible `DroneObservation` data only, including mission checklist, telemetry, visible site map, public assets, captured evidence artifacts, affordances, warnings, checklist status, reward breakdown, and latest action result. Hidden defects, true asset state, hidden weather details, and verifier evidence requirements are not exposed.
- Reward shape: dense shaping reward during collection, then terminal mission score after `submit_evidence_pack`, capped by safety and evidence-integrity gates. Fake photo IDs or unsupported report claims reduce the final score through the integrity gate.
- Default baseline tasks: `basic_thermal_survey`, `anomaly_confirmation`, and `audit_grade_strict_grounding`.

Submission commands:

```bash
python3.11 inference.py --policy scripted
python3.11 inference.py --task basic_thermal_survey --policy scripted
```

Docker and local server:

```bash
docker build -t dronecaptureops-gym .
docker run --rm -p 8000:8000 dronecaptureops-gym
curl -s -o /dev/null -w "%{http_code}" -X POST \
  http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{}'
```

Validation:

```bash
python3.11 -m pytest
openenv validate
bash -n scripts/validate-submission.sh
./scripts/validate-submission.sh http://127.0.0.1:8000 .
```

Training and notebook:

```bash
python -m training.generate_sft_data \
  --config training/configs/sft_default.yaml \
  --output artifacts/sft/sft-warmstart.jsonl

python -m training.sft_warmstart \
  --config training/configs/sft_train_default.yaml \
  --dataset artifacts/sft/sft-warmstart.jsonl
```

Judge notebook path: `training/colab_training_template.ipynb`.

Hugging Face Space deployment commands, once `HF_TOKEN` is available:

```bash
pip install -U huggingface_hub
hf auth login --token "$HF_TOKEN"
export SPACE_ID="<HF_USERNAME>/dronecaptureops-gym"
python - <<'PY'
import os
from huggingface_hub import HfApi

HfApi().create_repo(
    repo_id=os.environ["SPACE_ID"],
    repo_type="space",
    space_sdk="docker",
    exist_ok=True,
)
PY
hf upload "$SPACE_ID" . . --repo-type space
./scripts/validate-submission.sh "https://${SPACE_ID/\//-}.hf.space" .
```

## What Exists Now

- Deterministic `FastGeometrySim` style backend through `GeometryController`.
- Solar inspection MVP with rows `B4-B8`, a substation no-fly zone, seeded hidden defects, weather, and battery use.
- OpenEnv `reset`, `step`, and visible `state` support.
- High-level tool registry with schema validation.
- Safety wrapper before flight/gimbal actions.
- Structured RGB/thermal capture-quality metadata.
- Rich-sim scene serialization and a FastAPI-mounted browser console at `/ui/` for live state, action logs, captures, rewards, and comparison runs.
- A live-session API under `/live/*` for deterministic manual steps, event replay, and base/SFT/RL-style model comparison specs.
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
  rich_sim/      renderer-friendly scene/event serialization
  utils/         math, geo, logging, serialization helpers
server/          OpenEnv + live-session FastAPI entrypoint and static UI
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
dronecaptureops-server                                 # 5. start OpenEnv + live UI, then open http://localhost:8000/ui/
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

Available suites: `smoke`, `curriculum_easy`, `curriculum_medium`, `hard_eval`, `demo`, `demo_llm_inspection`, and `solar_tasks` (defined in `dronecaptureops/generation/suites.py`). Available policies: `random`, `weak_scripted`, `scripted`.

Trace output includes `episode_steps.json`, `evidence_log.json`, `route_log.json`, `inspection_report.json`, `trace.json`, and `trace.md`. Each step records the action, next observation, reward breakdown, reward deltas, warnings, and visible state changes so training runs can be debugged trajectory by trajectory.

## Scenario Suites vs. Task IDs

There are two orthogonal selectors and they do different things:

- **Task ID** (`env.reset(task="anomaly_confirmation")`, alias `task_id=`) picks a deterministic *mission variant* from `dronecaptureops/tasks/solar_tasks.py`. The solar tasks override hidden defects, weather, battery, step budget, extra zones/viewpoints, and quality thresholds. They are the unit of RL task-conditioning.
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

The same server also mounts the rich-sim console and live APIs:

```bash
open http://localhost:8000/ui/
```

In the console, use **AI mission run** to enter a detailed objective prompt,
choose a model/backend spec, and stream each model-selected tool call into the
scene and command log. Use **Dataset replay** to pick an SFT/rollout JSONL
record and render the stored assistant tool-call sequence through the same live
event path.

The AI mission panel defaults to HF router + `Qwen/Qwen3-4B-Instruct-2507`.
`server.app` loads `.env` automatically, so an `HF_TOKEN` in that file is enough
for the default run. Choose **OpenAI GPT-5.4 Mini** in the policy selector to
run `gpt-5.4-mini`; the server will use `OPENAI_API_KEY`, `OPENAI_KEY`,
`GPT_API_KEY`, or `GPT_TOKEN` from `.env`.

Useful live endpoints:

- `POST /live/sessions` — create/reset a live environment session.
- `POST /live/sessions/{session_id}/step` — execute one `RawDroneAction` and append replay events.
- `POST /live/sessions/{session_id}/run_model` — run one `ModelRunSpec` against the current live session, appending each model-selected tool call to the event log.
- `POST /live/sessions/{session_id}/replay` — replay assistant tool calls from an SFT JSONL record, rollout JSON, trace steps, or an explicit `actions` list.
- `GET /live/sessions/{session_id}/events` — replay session events with scene snapshots.
- `POST /live/compare` — run multiple model specs on the same task/seed prompt contract.
- `GET /live/tasks` and `GET /live/suites` — discover task IDs and suites for the UI.
- `GET /live/diagnostics/logs?limit=200` — inspect recent backend diagnostics.

Live-session diagnostics are also written as JSONL to
`artifacts/live/live-server.jsonl` by default. Override with
`DRONECAPTUREOPS_LIVE_LOG_PATH=/path/to/log.jsonl`.

Example model run:

```bash
curl -X POST http://localhost:8000/live/sessions/demo/run_model \
  -H 'Content-Type: application/json' \
  -d '{
    "task_id": "basic_thermal_survey",
    "max_steps": 8,
    "user_instruction": "Inspect the assigned solar rows, capture evidence, and return safely.",
    "spec": {"name": "base", "policy": "hf", "model": "Qwen/Qwen3-4B-Instruct-2507"}
  }'
```

Example SFT/trajectory replay:

```bash
curl -X POST http://localhost:8000/live/sessions/replay-demo/replay \
  -H 'Content-Type: application/json' \
  -d '{"source_path": "artifacts/sft/sft-warmstart.jsonl", "record_index": 0}'
```

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

## Controller Strategy

The default backend is `GeometryController` (deterministic, CPU-light). `DroneKitSITLController` exists as a placeholder that raises `NotImplementedError`; SITL/ArduPilot work is deferred until the reward and benchmark surface stabilizes. Details and the minimum viable SITL feature set are in [docs/controller-strategy.md](docs/controller-strategy.md).

## Packaging Notes

Runtime dependencies (`openenv-core`, `pydantic`, `pyyaml`, `fastapi`, `uvicorn`) are pinned in `pyproject.toml` so a clean `pip install -e .` boots the OpenEnv server without surprises. `setuptools.packages.find` includes only `dronecaptureops*` and `server*`; `tests/`, `training/`, `examples/`, `docs/`, and `artifacts/` are excluded from the wheel. `uv.lock` is checked in for reproducible local installs but is not the canonical lock — `pyproject.toml` is. A `LICENSE` file has not been added yet; if this repo is going public, drop one in at the root before the first release.

## Demo Direction

A weak baseline should fly to a generic overview point, capture one or two low-value photos, miss an anomaly or close-up, and submit an incomplete pack. A stronger trained agent should read the checklist, capture thermal overview, inspect quality, reposition for missing rows or anomalies, capture RGB close-up evidence, avoid the no-fly zone, return home, land, and submit a photo-linked evidence pack.
