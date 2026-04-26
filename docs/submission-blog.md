# DroneCaptureOps Gym: Training LLMs To Act As Drone Inspection Directors

## TL;DR

DroneCaptureOps Gym is an OpenEnv-compatible RL environment where the agent
acts as an aerial **inspection director**. It does not control motors. It
issues high-level tool calls — mission review, safe waypoint flight,
gimbal/camera control, RGB and thermal capture, capture inspection,
return-home, and final evidence-pack submission — and is graded on whether
the evidence it actually collected supports the report it actually filed.
The goal is to train LLMs into reliable drone operating-system agents that
can plan, capture grounded evidence, stay safe, and report honestly.

---

## Why This Problem Matters

Solar farms, bridges, construction sites, industrial plants, disaster zones,
and perimeter patrols all need recurring aerial inspection. Today a human
operator (or a team) decides where to fly, what to capture, when evidence is
"enough," when to recapture, and how to write the final report. Those
decisions are the hard part — autopilot already handles flight.

LLM agents need environments that exercise that grounded operational
judgment, not toy game behavior. DroneCaptureOps focuses on:

- deciding what evidence is missing,
- deciding where to capture next,
- deciding when to recapture vs. move on,
- avoiding no-fly zones and battery dead-ends,
- and only citing evidence the agent actually has.

If a model can do this consistently across deterministic tasks, the same
behavior transfers cleanly to a real inspection workflow.

---

## What The Environment Simulates

- OpenEnv-compatible RL environment (`spec_version: 1`,
  `app: server.app:app`).
- Current fleshed-out domain: **solar farm inspection**. `bridge`,
  `construction`, and `industrial` builders exist as placeholders.
- Backend: deterministic geometry simulator via `GeometryController`. A
  `DroneKitSITLController` adapter exists as a placeholder for future
  ArduPilot/SITL work.
- Agent role: **inspection director**, not a flight controller.
- Tool surface: mission/map, flight, camera/gimbal, evidence/report.
- Hidden vs. visible state is enforced as an invariant: hidden defects,
  true asset state, and verifier-only labels live only in `EpisodeWorld`
  and never appear in observations.

The benchmark is **active visual inspection** — collecting grounded
evidence — not shortest-path navigation.

---

## Action Space

Actions are structured tool calls. The public surface is built from a
single `ToolRegistry`, so validation and availability come from one place.

Categories:

- **Mission/map**: `get_site_map`, `get_mission_checklist`,
  `get_telemetry`, `list_assets`, `estimate_view`,
  `estimate_return_margin`, `request_route_replan`
- **Flight**: `takeoff`, `fly_to_viewpoint`, `move_to_asset`, `hover`,
  `return_home`, `land`
- **Camera/gimbal**: `set_gimbal`, `set_zoom`, `set_camera_source`,
  `point_camera_at`, `capture_rgb`, `capture_thermal`, `inspect_capture`
- **Evidence/report**: `mark_target_inspected`, `submit_evidence_pack`

Example action:

```json
{
  "tool_name": "fly_to_viewpoint",
  "arguments": {
    "x": 30,
    "y": 24,
    "z": 22,
    "yaw_deg": -90,
    "speed_mps": 5
  }
}
```

---

## Observation Space

Observations include:

- visible mission instructions and task-conditioned objective,
- telemetry (pose, velocity, gimbal, camera state),
- battery and weather (visible portions only),
- site map and visible assets,
- airspace zones (no-fly / restricted),
- capture log and last capture (with quality metadata),
- evidence artifacts (real captured photo IDs),
- checklist status,
- reward breakdown (including `debug` payload distinguishing shaping vs.
  outcome reward).

What observations **do not** include: hidden defects, true asset state,
hidden weather details, obstacle schedules, and verifier evidence
requirements. `tests/test_no_hidden_state_leakage.py` enforces this.

---

## Tasks And Difficulty

Three default baseline tasks span easy → medium → hard:

- **Easy**: `basic_thermal_survey` — fly the rows, capture thermal
  overview, submit a clean evidence pack.
- **Medium**: `anomaly_confirmation` — find and confirm a thermal anomaly
  with a corresponding RGB close-up.
- **Hard**: `audit_grade_strict_grounding` — strict integrity gate;
  reports must cite real captured artifact IDs, with no unsupported
  claims.

The `solar_tasks` catalog (`dronecaptureops/tasks/solar_tasks.py`) defines
30+ deterministic, programmatically graded mission variants — including
low-battery, obstacle detour, privacy zones, edge-row quality bars, glare
artifacts, and severity-weighted triage. Tasks are the unit of RL
task-conditioning. Scenario suites
(`smoke`, `curriculum_easy`, `curriculum_medium`, `hard_eval`, `demo`,
`solar_tasks`) are the unit of benchmark/regression reporting.

---

## Reward Design

Reward is published every step but its meaning depends on whether the agent
has called `submit_evidence_pack`:

- **Before submission** — `total` is a small *shaping* reward derived from
  captured progress; outcome scoring is suppressed so the agent cannot farm
  reward by hovering and capturing forever.
- **After submission** — `total` is the actual mission outcome score using
  *cited* evidence, with safety and integrity caps applied. `done=True`.

Terminal formula:

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

`safety_gate` and `integrity_gate` are **caps, not multipliers**. A no-fly
violation caps total at ~0.10. Citing fake or unsupported photo IDs caps
via integrity. `process_reward` is bounded at 0.10 so dense shaping never
dominates the outcome. This pushes the model toward grounded, safe
behavior rather than reward farming.

---

## OpenEnv Compliance

- `openenv.yaml` declares `name: dronecaptureops-gym`,
  `runtime: fastapi`, `app: server.app:app`, `port: 8000`.
- `server.app:app` exposes the OpenEnv `reset` / `step` / `state`
  endpoints, plus a live-session API under `/live/*` and the rich-sim
  console at `/ui/`.
- Repo contains a root `inference.py` that runs an episode end-to-end
  with any of `scripted`, `random`, `openai`, `anthropic`, `hf` policies
  using a shared `RolloutRunner`.
- Hugging Face Space deployment: TODO_HF_SPACE_URL
- Root `Dockerfile`: not yet in the repo root — will be added as part of
  the submission package.
- `openenv validate` status: TODO — will be re-run and recorded after the
  Dockerfile and Space are in place.

---

## Inference And Baseline

Run the scripted baseline locally:

```bash
python inference.py --task basic_thermal_survey --policy scripted
```

The CLI also supports `--policy openai`, `--policy anthropic`, and
`--policy hf` (with `--model`, `--api-base-url`, `--api-key`,
`--temperature`, `--max-tokens`). Output is a JSON summary with
`task_id`, `policy`, `seed`, `steps`, `total_reward`, `success`,
`anomalies_detected`, `anomaly_rgb_pairs`, and remaining battery.

For full per-step traces during evaluation:

```bash
python -m training.run_suite --suite smoke --policy scripted
python -m training.trace_episode --suite demo --episode-index 0 \
  --policy scripted --output-dir artifacts/trace-demo
```

---

## Training Setup

The training stack is TRL-based and runs on either local GPUs or HF Jobs:

1. Generate SFT warm-start data from scripted/teacher rollouts.
2. SFT-warm-start a small instruct model (Qwen3-4B-Instruct-2507 in our
   reference run) into the tool-call format.
3. Continue with PPO (TRL) on the live environment with KL anchoring to
   the SFT adapter.

Commands:

```bash
python -m training.generate_sft_data \
  --config training/configs/sft_default.yaml \
  --output artifacts/sft/sft-warmstart.jsonl
```

```bash
python -m training.sft_warmstart \
  --config training/configs/sft_train_default.yaml \
  --dataset artifacts/sft/sft-warmstart.jsonl \
  --output-dir artifacts/sft-checkpoints
```

```bash
python -m training.train_ppo \
  --config training/configs/ppo_train_default.yaml
```

W&B tracking is wired into both SFT and PPO trainers.

---

## Training Results

Real artifacts that exist in this repo today:

- SFT warm-start dataset committed to artifacts: `artifacts/sft/sft-warmstart.jsonl`
  (~60 MB, generated from scripted rollouts) and held-out split.
- Two PPO sweep runs configured and launched on 2026-04-26 from the SFT
  adapter (`adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint`,
  eval_loss 0.300 at step 40 / epoch 0.74). Sweep dimension: `kl_coef`
  at the informative extremes (0.02 vs. 0.005). See
  `artifacts/ppo-runs-2026-04-26/README.md` for the full sweep design.

What is **not** yet finalized for submission:

- TODO: final loss / reward plots from a completed PPO run.
- TODO: end-to-end eval table comparing scripted, SFT-warm-start, and
  PPO-finetuned policies on the three baseline tasks.

Pending links to be filled in once a tracked run completes:

- `TODO_EXPERIMENT_TRACKING_URL`
- `TODO_LOSS_PLOT_URL`
- `TODO_REWARD_PLOT_URL`
- `TODO_TRAINED_MODEL_OR_ADAPTER_URL`

No fabricated metrics are claimed here. Numbers will only appear after
they exist in a real tracker run.

---

## How Judges Can Run It

Local end-to-end:

```bash
pip install -e ".[dev]"
pytest                                              # fast deterministic suite
python inference.py --task basic_thermal_survey --policy scripted
python -m training.run_suite --suite smoke --policy scripted
dronecaptureops-server                              # OpenEnv + live UI on :8000
```

Containerized (after the root Dockerfile lands):

```bash
docker build -t dronecaptureops-gym .
docker run --rm -p 8000:8000 dronecaptureops-gym
```

Submission validator (after `scripts/validate-submission.sh` lands):

```bash
./scripts/validate-submission.sh TODO_HF_SPACE_URL .
```

Notebook for judges:

- `training/colab_training_template.ipynb` — TODO, will be added as part
  of the submission package.
- `TODO_EXECUTED_TRAINING_NOTEBOOK_URL`

---

## Demo / Video / Extra Materials

- `TODO_WRITEUP_OR_VIDEO_URL`
- `TODO_OPTIONAL_PRESENTATION_URL`

Large video files are not committed to this repo or any HF Space. Public
URLs will be linked when available.

---

## What Makes This Different

- The agent must collect evidence, not just output an answer. Reports
  reference real captured photo IDs.
- Visible state vs. hidden verifier state is a load-bearing invariant
  protected by tests, not a loose convention.
- Reward punishes unsafe and ungrounded behavior via caps, not soft
  weights. A no-fly violation or fake citation cannot be averaged away.
- The tool interface mirrors real drone operations (telemetry,
  gimbal/zoom, return-home, evidence pack), so behavior the model learns
  here is closer to deployable than typical gridworld RL.
- The architecture is multi-domain by design (solar today; bridge,
  construction, industrial scaffolded).

---

## Current Limitations

- Geometry-first simulator. No photorealistic rendering yet.
- `DroneKitSITLController` is a placeholder; ArduPilot/SITL integration
  is deferred until the reward and benchmark surface stabilize.
- Bridge / construction / industrial domains exist as builders but are
  not yet populated with assets, tasks, or rewards.
- Submission packaging artifacts (root `Dockerfile`, Space deployment,
  `scripts/validate-submission.sh`, judge notebook) are still pending in
  the repo as of this writing.
- Final RL training plots are pending a completed tracked run; no
  fabricated numbers are reported here.

---

## Conclusion

DroneCaptureOps Gym is a step toward training LLMs as trustworthy drone
operating-system agents. The core benchmark is grounded inspection: plan,
capture, verify, report, and stay safe. Tasks are deterministic, rewards
are grounded in real captures, and the architecture leaves room to grow
from solar into bridge, construction, industrial, disaster response, and
security patrol domains without rewriting the agent contract.
