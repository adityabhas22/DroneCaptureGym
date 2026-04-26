# Teaching an LLM To Direct A Drone Inspection: DroneCaptureOps Gym

> One-line pitch: DroneCaptureOps Gym is an OpenEnv-compatible reinforcement learning environment that turns an LLM into a high-level drone inspection director, where the only way to score is to gather grounded visual evidence.

This is our submission to the OpenEnv India Hackathon 2026.

- Hugging Face Space: `TODO_HF_SPACE_URL`
- Training Run Notebook: `TODO_EXECUTED_TRAINING_NOTEBOOK_URL`
- Demo video (if recorded): `TODO_WRITEUP_OR_VIDEO_URL`

## TL;DR

Most LLM agent benchmarks reward answering from a prompt or solving a closed puzzle. Real field work is different: you have to gather evidence before you can answer, and a confident report is only valuable if it is grounded in the data you actually collected.

DroneCaptureOps Gym makes that capability trainable. The agent is the inspection director for a solar farm site. It plans safe viewpoints, captures thermal and RGB evidence, inspects photo quality, respects no-fly zones, returns home, and submits a final evidence pack tied to real photo IDs. If it lies about evidence or violates safety, the score is capped.

## Why This Problem

The world is full of inspection work that LLMs cannot yet do well: power infrastructure, telecom towers, wind farms, bridges, construction sites, agriculture. These tasks are not solved by chain-of-thought alone. They require:

- Knowing what evidence is missing.
- Choosing the next safe action that produces a useful measurement.
- Verifying capture quality before moving on.
- Refusing to claim findings that are not supported by captured data.

Today, even strong LLMs hallucinate confident reports when they have not been forced to ground their answers in observations. We wanted an environment that punishes that pattern and rewards the opposite: gather first, answer last.

DroneCaptureOps focuses on aerial solar inspection because it has a deep, naturally rich reward surface (thermal coverage, anomaly confirmation, route safety, evidence integrity) without requiring a heavy photorealistic simulator. The benchmark is designed to be ambitious in scope, but cheap and deterministic to train against.

## What The Agent Sees And Does

The agent acts through structured OpenEnv tool calls. There is no raw motor control. The tool surface is split into four families:

- Mission and map tools: `get_mission_checklist`, `get_site_map`, `get_telemetry`, `list_assets`, `estimate_view`, `estimate_return_margin`, `request_route_replan`.
- Flight tools: `takeoff`, `fly_to_viewpoint`, `move_to_asset`, `hover`, `return_home`, `land`.
- Camera tools: `set_gimbal`, `set_zoom`, `set_camera_source`, `point_camera_at`, `capture_rgb`, `capture_thermal`, `inspect_capture`.
- Evidence tools: `mark_target_inspected`, `submit_evidence_pack`.

A typical successful trajectory looks like:

1. Read the mission checklist and site map.
2. Take off and fly to safe standoff viewpoints.
3. Capture thermal coverage across the required solar rows.
4. Inspect captures and reposition if quality or coverage is weak.
5. Capture RGB context for any suspected anomaly.
6. Return home, land, and submit a photo-linked evidence pack.

Every observation includes telemetry, mission checklist progress, capture log entries with quality metadata, the dynamic available-tool catalog, and a dense reward breakdown. Hidden verifier state (true asset state, hidden defects, hidden weather details) never leaks into the observation. We have a dedicated test for that invariant.

A real action looks like this:

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

## Reward Design: Hard To Game By Construction

The reward is composable. After `submit_evidence_pack` is called, the terminal score is:

```text
total = clamp(min(safety_gate, integrity_gate,
                  0.45 * evidence_success
                + 0.20 * required_coverage
                + 0.15 * issue_capture
                + 0.10 * operational_efficiency
                + 0.10 * grounded_report
                + process_reward
                - penalties),
              -1, 1)
```

A few things are deliberate:

- `safety_gate` and `integrity_gate` are caps, not multipliers. A no-fly violation caps the total at roughly `0.10`. A report citing fake photo IDs or unsupported claims is capped by the integrity gate.
- `process_reward` is bounded at `0.10` so the agent gets dense feedback while it is still flying, but cannot stack shaping rewards by hovering and capturing forever.
- The components mirror the legacy fields (`target_coverage`, `defect_visibility`, `route_efficiency`, `report_grounding`) for backward compatibility.
- Verifier checks live in `rewards/verifiers.py` and use simulator state plus capture metadata only. There is no LLM judge in the loop.

This means an agent cannot win by being eloquent. It has to actually fly safely, capture the right photos, and submit a report that is consistent with what was captured.

## What Makes This Original

Hackathon judges have seen a lot of grid worlds, chess clones, and text puzzles. We optimized for three things they care about:

1. Domain originality. We are training an LLM to direct an aerial visual inspection mission. The domain is realistic, has clear value, and is underexplored in RL/LLM training.
2. Reward signal that teaches. The reward decomposes into coverage, issue capture, operational efficiency, grounded reporting, capture quality, checklist completion, battery management, safety, and integrity. The signal is dense enough to learn from but capped where gaming is easy.
3. Anti-hallucination by construction. The benchmark reward cannot be earned by writing a confident report. Reports that cite photos that do not exist are capped by the integrity gate. This is, intentionally, a way to force grounded behavior at training time.

We want this environment to be useful as an open benchmark for any team that wants to push LLM agents toward grounded, safety-aware, evidence-driven decision making.

## Engineering: OpenEnv The Right Way

DroneCaptureOps is built on top of OpenEnv, not on top of a custom RL stack:

- Standard Gym/OpenEnv API: `reset`, `step`, and visible `state` are implemented on `DroneCaptureOpsEnvironment`.
- Typed Pydantic models for actions, observations, and visible state in `dronecaptureops/core/models.py`.
- Public tool registry with schema validation in `dronecaptureops/tools/__init__.py`. None of the public tool names collide with reserved MCP names like `reset`, `step`, `state`, or `close`.
- FastAPI server entrypoint in `server/app.py` so the environment runs unchanged on a Hugging Face Space (`spec_version: 1`, `runtime: fastapi`, `app: server.app:app`, `port: 8000`).
- Backend-neutral controller: `GeometryController` is the deterministic default; `DroneKitSITLController` exists as a placeholder boundary so the same command layer can later drive ArduPilot/DroneKit SITL.

The environment is fast: the entire test suite (334 tests) runs in about a minute on a laptop. That is on purpose. Slow benchmarks ruin RL iteration.

## Local Validation We Already Ran

We did not just claim it works. From the repo root:

- `pytest`: 334 tests passed.
- `python inference.py --policy scripted`: ran all three default tasks (`basic_thermal_survey`, `anomaly_confirmation`, `audit_grade_strict_grounding`) and printed `[START]`, `[STEP]`, and `[END]` records with `score=1.00` per task.
- `openenv validate`: returned `[OK] DroneCaptureGym: Ready for multi-mode deployment`.
- Local OpenEnv server `/reset`: returned HTTP 200 against `http://127.0.0.1:8000/reset`.

The submission validator script lives in `scripts/validate-submission.sh` and is a copy of the official OpenEnv pre-submission validator, so the readiness gate is reproducible.

## Tasks And Difficulty

The default `inference.py` baseline runs three deterministic task variants:

- Easy: `basic_thermal_survey`. Full thermal coverage of the required rows, safe route, return, land, and a grounded report.
- Medium: `anomaly_confirmation`. Detect a hidden thermal anomaly and capture target-specific RGB context.
- Hard: `audit_grade_strict_grounding`. Multi-issue evidence and a stricter audit-grade evidence pack.

The full `solar_tasks` suite contains a much larger task-conditioned benchmark for training and evaluation. Suites like `smoke`, `curriculum_easy`, `curriculum_medium`, and `hard_eval` exist for regression reporting.

## Training Setup

The repo includes a real TRL-based SFT warm-start path (`training/sft_warmstart.py`) plus the data generation, evaluation, plotting, and HF Jobs scaffolding to run end-to-end training. The training notebook (linked at the top of this post) shows the exact path used for the submission run.

Important honesty disclaimer: any plots and numbers in this post must come from a real run. We do not fabricate evidence. The training notebook URL above is the source of truth; please follow it to reproduce.

If you want to run the same path locally:

```bash
# generate warm-start data
python -m training.generate_sft_data \
  --config training/configs/sft_default.yaml \
  --output artifacts/sft/sft-warmstart.jsonl

# train a small model with TRL SFT
python -m training.sft_warmstart \
  --config training/configs/sft_train_default.yaml \
  --dataset artifacts/sft/sft-warmstart.jsonl \
  --output-dir artifacts/sft-checkpoints

# plot real metrics from the run
python -m training.plot_training_metrics \
  --trainer-log artifacts/sft-checkpoints/metrics/trainer_log_history.json \
  --output-dir artifacts/submission-plots
```

## Results

The full results, plots, and tracker links are attached to the training notebook URL above and to the README in the Hugging Face Space.

Until those are filled in below, this section is intentionally empty. We refuse to publish placeholder numbers as if they were real.

- Loss curve: `TODO_LOSS_PLOT_URL`
- Reward/eval curve: `TODO_REWARD_PLOT_URL`
- Tracker run: `TODO_EXPERIMENT_TRACKING_URL`
- Trained adapter/model: `TODO_TRAINED_MODEL_OR_ADAPTER_URL`
- One-paragraph results summary: `TODO_RESULTS_SUMMARY`

## How To Try It Yourself

Once the Hugging Face Space is live, you can hit it directly:

```bash
curl -X POST <HF_SPACE_URL>/reset -H "Content-Type: application/json" -d '{}'
```

To run the environment locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python inference.py --policy scripted
```

You should see structured `[START]`, `[STEP]`, and `[END]` lines for each task, with grounded reward scores in `[0.000, 1.000]`.

To run the OpenEnv server locally:

```bash
dronecaptureops-server
# or
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

If you want to use a model-backed policy instead of the scripted baseline, set:

```bash
export API_BASE_URL="https://your-openai-compatible-endpoint/v1"
export MODEL_NAME="your-model-id"
export HF_TOKEN="your-api-token"
python inference.py
```

## What Comes Next

This is not the end-state. We deliberately shipped a focused, real-world MVP and made the controller, scenario, and reward boundaries clean enough to extend without rewrites:

- Promote `DroneKitSITLController` from placeholder to a real ArduPilot/DroneKit SITL adapter so the same OpenEnv tool calls drive a fully simulated autopilot stack.
- Add additional inspection domains (bridge, telecom tower, wind farm) as new `DomainScenarioBuilder` subclasses.
- Add reward components for human-in-the-loop verification when the agent is uncertain.
- Build a curriculum that scales from `curriculum_easy` to `hard_eval` and beyond.
- Use the same evidence-grounded benchmark to study hallucination behavior in long-horizon tool-using LLM agents.

## Why This Matters

LLM agents will eventually have to operate physical systems where every action has a cost and every report has consequences. Drone inspection is a clean, high-value domain to study that capability today, without burning hardware or risking people.

DroneCaptureOps Gym is our bet that the right next benchmark is not another puzzle. It is a real-world directed-inspection environment where the agent has to gather grounded evidence to win.

If you are training models, ablation-testing reward shaping, or looking for a non-game OpenEnv benchmark with depth, we built this for you.

## Links

- Hugging Face Space (run the environment): `TODO_HF_SPACE_URL`
- Training Run Notebook: `TODO_EXECUTED_TRAINING_NOTEBOOK_URL`
- Demo video (optional): `TODO_WRITEUP_OR_VIDEO_URL`
- Repository README: see [README.md](README.md)
- Reward and reward integrity details: see [README.md](README.md)
- Submission readiness checklist: see [docs/hackathon-submission-readiness-checklist.md](docs/hackathon-submission-readiness-checklist.md)
- Submission video script and DroneKit/SITL guide: see [docs/submission-video-script.md](docs/submission-video-script.md)

## Acknowledgements

Built for the OpenEnv India Hackathon 2026 on top of OpenEnv (`openenv-core`), Pydantic, FastAPI, TRL, Hugging Face Hub, and inspiration from the DroneKit/ArduPilot SITL ecosystem.
