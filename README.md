# DroneCaptureOps Gym

DroneCaptureOps Gym is an OpenEnv-compatible environment for training LLM agents to operate as aerial inspection directors. The agent does not control motors. It acts through high-level tools for mission review, safe waypoint flight, gimbal/camera control, RGB and thermal capture, capture inspection, return-home, and final evidence-pack submission.

The benchmark focus is active visual inspection: deciding what evidence is missing, where to capture next, when to recapture, and whether the final evidence pack is grounded in real photos.

## Why This Exists

Most LLM agent benchmarks reward answering from a prompt or solving a closed puzzle. Real inspection work is different: the agent has to gather evidence before it can answer, and a confident report is only valuable if it is grounded in the data it actually collected.

DroneCaptureOps turns that capability gap into a trainable OpenEnv task. The agent is responsible for directing a drone inspection of a solar site: it must read the mission, plan safe viewpoints, collect thermal overview evidence, capture RGB context for issues, inspect photo quality, avoid no-fly constraints, return home, and submit a final evidence pack with real photo IDs.

The goal is not shortest-path navigation. The goal is evidence-grounded field reasoning: knowing when the current information is not enough, taking the next useful measurement, and refusing to hallucinate unsupported findings.

## How The Environment Works

Each episode starts from a deterministic solar inspection task. The hidden verifier knows the true asset state, hidden defects, weather details, and evidence requirements, but the agent only sees public mission state: telemetry, map geometry, available tools, visible assets, capture metadata, checklist progress, and reward breakdowns.

The agent acts through structured OpenEnv tool calls. A typical successful trajectory looks like:

1. Review the mission checklist and site map.
2. Take off and fly to safe standoff viewpoints.
3. Capture thermal coverage across required solar rows.
4. Inspect captures and reposition if quality or coverage is weak.
5. Capture RGB context for suspected anomalies.
6. Return home, land, and submit a photo-linked evidence pack.

The environment grades the whole mission, not just the final text. Reports that cite fake photo IDs or claim unsupported issues are capped by the integrity gate, and unsafe behavior is capped by the safety gate.

## Agent Actions And Rewards

The agent has four broad action families:

- Mission/map tools: inspect the checklist, map, telemetry, assets, return margin, and route alternatives.
- Flight tools: take off, fly to viewpoints, move near assets, hover, return home, and land.
- Camera tools: set gimbal/zoom/source, point at assets, capture RGB/thermal photos, and inspect capture metadata.
- Evidence tools: mark targets inspected and submit the final evidence pack.

Rewards are dense enough to teach process but strict enough to prevent gaming. Before submission, the agent receives bounded shaping feedback for useful progress such as coverage, capture quality, checklist completion, safety, and battery management. After `submit_evidence_pack`, the terminal reward scores the grounded outcome: required evidence, issue capture, operational efficiency, report quality, safety, and integrity.

## Try It

Public Hugging Face Space: `TODO_HF_SPACE_URL`

After the Space is live, sanity-check it with:

```bash
curl -X POST TODO_HF_SPACE_URL/reset -H "Content-Type: application/json" -d '{}'
```

To run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python inference.py --policy scripted
```

The scripted baseline runs three deterministic tasks and emits submission-style `[START]`, `[STEP]`, and `[END]` records.

## Results And Submission Materials

Final public links will be added before submission:

- Hugging Face Space: `TODO_HF_SPACE_URL`
- Training/eval results: `TODO_RESULTS_SUMMARY`
- Loss curve from a real run: `TODO_LOSS_PLOT_URL`
- Reward/eval curve from a real run: `TODO_REWARD_PLOT_URL`
- Training tracker run, TensorBoard/W&B/etc.: `TODO_EXPERIMENT_TRACKING_URL`
- Executed training notebook: `TODO_EXECUTED_TRAINING_NOTEBOOK_URL`
- Trained adapter/model artifact: `TODO_TRAINED_MODEL_OR_ADAPTER_URL`
- Short writeup, mini-blog, or under-2-minute video: `TODO_WRITEUP_OR_VIDEO_URL`
- Optional slides/demo page: `TODO_OPTIONAL_PRESENTATION_URL`

The video/storyboard draft lives in [docs/submission-video-script.md](docs/submission-video-script.md). The visual system flow lives in [docs/submission-system-flow.md](docs/submission-system-flow.md), with a shareable Mermaid source at [docs/submission-hero-flow.mmd](docs/submission-hero-flow.mmd) and sharing instructions in [docs/submission-visualization-share.md](docs/submission-visualization-share.md). The submission blog post lives at [BLOG.md](BLOG.md) and is the source for the form's "Blog Post" field once this repo is pushed to a Hugging Face Space. Final plots and claims must come from a real run, not placeholders.

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
training/        TRL warm-start, plotting, HF Jobs, and eval scaffolds
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

Available suites: `smoke`, `curriculum_easy`, `curriculum_medium`, `hard_eval`, `demo`, `solar_tasks` (defined in `dronecaptureops/generation/suites.py`). Available policies: `random`, `weak_scripted`, `scripted`.

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

## Hackathon Submission Status

This repo is prepared as a real-world OpenEnv environment for aerial solar inspection, not a game/toy task. It exposes a FastAPI OpenEnv server through `server.app:app`, typed Pydantic action/observation/state models, deterministic task-conditioned resets, shaped rewards, and a root `inference.py` baseline runner.

Full readiness is tracked in [docs/hackathon-submission-readiness-checklist.md](docs/hackathon-submission-readiness-checklist.md). The checklist is intentionally explicit about what is already real and what must still be filled after deployment/training.

### Public Submission Links

Fill these before final submission. Do not replace them with placeholders in the final submitted README.

- Hugging Face Space: `TODO_HF_SPACE_URL`
- Results summary: `TODO_RESULTS_SUMMARY`
- Executed Colab / training notebook: `TODO_EXECUTED_TRAINING_NOTEBOOK_URL`
- Training tracker run, TensorBoard/W&B/etc.: `TODO_EXPERIMENT_TRACKING_URL`
- Loss curve from a real run: `TODO_LOSS_PLOT_URL`
- Reward/eval curve from a real run: `TODO_REWARD_PLOT_URL`
- Trained adapter/model artifact: `TODO_TRAINED_MODEL_OR_ADAPTER_URL`
- Short writeup, mini-blog, or under-2-minute video: `TODO_WRITEUP_OR_VIDEO_URL`
- Optional slides/demo page: `TODO_OPTIONAL_PRESENTATION_URL`

Current local status:

- `DONE`: OpenEnv package/server/model structure, deterministic tasks, reward shaping, root `inference.py`, Dockerfile presence, README environment docs.
- `PARTIAL`: training infrastructure. The repo has a working TRL SFT warm-start path, tracking config, HF Jobs wrapper, and plotting utility, but `training/train_grpo.py` is still a scaffold and there is no completed RL run checked into the repo.
- `PENDING`: HF Space deployment, real training run, real plots, public writeup/video, and public artifact links.
- `BLOCKED LOCALLY`: Docker build validation requires Docker Desktop/daemon to be running.

### Submission Checklist

| Requirement | Status Now | Evidence / File | Final Action Needed |
| --- | --- | --- | --- |
| Use OpenEnv latest release/framework | Done | `pyproject.toml`, `openenv.yaml`, `server/app.py`; `openenv validate` passed locally. | Re-run `openenv validate` before submission. |
| Real-world non-game task | Done | Solar farm aerial inspection with evidence capture, safety, and reporting. | None. |
| Typed observation/action/state models | Done | `dronecaptureops/core/models.py`, `server/app.py`. | None. |
| Minimum 3 tasks | Done | `basic_thermal_survey`, `anomaly_confirmation`, `audit_grade_strict_grounding`. | Keep these as easy/medium/hard defaults in `inference.py`. |
| Meaningful reward with partial progress | Done | `dronecaptureops/rewards/`, README reward section, reward tests. | None. |
| Baseline inference script | Done | Root `inference.py`; emits `[START]`, `[STEP]`, `[END]`. | Validate against final evaluator. |
| OpenAI-compatible inference env vars | Done | `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` supported by `inference.py`. | Set real values for model-backed baseline. |
| Dockerfile | Present, not locally verified | `Dockerfile` exists. | Start Docker and run `./scripts/validate-submission.sh <HF_SPACE_URL> .`. |
| HF Space deployment | Pending | Placeholder above. | Push the environment to a public Hugging Face Space and add URL. |
| README links to HF Space | Pending | Placeholder above. | Replace `TODO_HF_SPACE_URL`. |
| Working training script | Partial | `training/sft_warmstart.py` is real TRL SFT; `training/train_grpo.py` is not real RL yet. | Implement/run actual RL training or clearly submit SFT+eval if accepted. |
| Colab notebook | Partial | [training/colab_training_template.md](training/colab_training_template.md) is a Colab-ready runbook. | Convert/run as an actual Colab notebook and link executed notebook. |
| Experimental tracking | Prepared | `training/configs/sft_train_default.yaml` reports to TensorBoard. | Run training and publish TensorBoard/W&B evidence. |
| Loss/reward plots | Prepared, no evidence yet | `training/plot_training_metrics.py` plots real logs only. | Run real training/eval and link generated plots. |
| Short writeup/video | Prepared locally, pending public upload | [docs/submission-video-script.md](docs/submission-video-script.md) plus placeholder above. | Record/publish public writeup/video and link it. |
| No large video files in repo/Space | Done currently | No media artifacts are committed. | Keep videos external via public URLs. |

### Submission Validation

Run the submission-style baseline:

```bash
python inference.py
```

By default, this runs three tasks and emits structured stdout lines:

```text
[START] task=basic_thermal_survey env=dronecaptureops-gym model=scripted
[STEP] step=1 action={"arguments":{},"tool_name":"get_mission_checklist"} reward=0.00 done=false error=null
[END] success=true steps=24 score=1.00 rewards=0.00,0.03,...
```

If `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` are set, the default policy uses the OpenAI client against that OpenAI-compatible endpoint. Without credentials, it falls back to the deterministic scripted baseline so local validation remains reproducible.

Required environment variables for model-backed inference:

```bash
export API_BASE_URL="https://your-openai-compatible-endpoint/v1"
export MODEL_NAME="your-model-id"
export HF_TOKEN="your-api-token"
```

Build and run the container:

```bash
docker build -t dronecaptureops-gym .
docker run --rm -p 8000:8000 dronecaptureops-gym
```

Validate the OpenEnv metadata/server locally:

```bash
openenv validate
```

Run the copied pre-submission validator after the HF Space exists:

```bash
./scripts/validate-submission.sh TODO_HF_SPACE_URL .
```

The validator checks `/reset`, Docker build, and `openenv validate`. It cannot pass with the placeholder URL, and Docker build cannot pass unless Docker is running.

### Training And Evidence

Generate warm-start trajectories:

```bash
python -m training.generate_sft_data \
  --config training/configs/sft_default.yaml \
  --output artifacts/sft/sft-warmstart.jsonl
```

Run the TRL SFT trainer:

```bash
python -m training.sft_warmstart \
  --config training/configs/sft_train_default.yaml \
  --dataset artifacts/sft/sft-warmstart.jsonl \
  --output-dir artifacts/sft-checkpoints
```

Open local TensorBoard tracking:

```bash
tensorboard --logdir artifacts/sft-checkpoints
```

Plot real logs after a completed run:

```bash
python -m training.plot_training_metrics \
  --trainer-log artifacts/sft-checkpoints/metrics/trainer_log_history.json \
  --output-dir artifacts/submission-plots
```

If evaluation JSONL with rewards exists, include it:

```bash
python -m training.plot_training_metrics \
  --trainer-log artifacts/sft-checkpoints/metrics/trainer_log_history.json \
  --eval-jsonl artifacts/eval/model-eval.jsonl \
  --output-dir artifacts/submission-plots
```

Important: do not treat the commands above as completed training evidence. They are the prepared path. Final evidence must come from a real run and must be linked in the public submission links section.

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

## Observation Space

Each observation is a typed `DroneObservation` with:

- `system_message`, `error`, `done`, and scalar `reward`
- `available_tools` and a full `tool_catalog`
- visible `telemetry` with pose, battery, gimbal, camera, weather band, and autopilot status
- visible `mission` instructions, success criteria, task tags, row requirements, quality thresholds, and battery reserve
- visible `site_map` with assets, airspace zones, and viewpoints
- `visible_assets`, `evidence_artifacts`, `capture_log`, and `last_capture`
- `checklist_status` for thermal coverage, detected anomalies, RGB pairings, return/land/submission status
- dense `reward_breakdown` with progress, safety, integrity, efficiency, and grounding components

Hidden verifier state such as actual defects and true asset state is never exposed in observations or visible state.

## Baseline Tasks

The root `inference.py` baseline runs three deterministic tasks by default:

- Easy: `basic_thermal_survey` — full thermal coverage, safe route, return, land, grounded report.
- Medium: `anomaly_confirmation` — detect one hidden thermal anomaly and capture target-specific RGB context.
- Hard: `audit_grade_strict_grounding` — collect multi-issue evidence and submit a stricter auditable evidence pack.

Scores are emitted in `[END]` lines as values in `[0.000, 1.000]`. The full `solar_tasks` suite contains the larger task-conditioned benchmark for training/evaluation.

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
