# DroneCaptureOps Submission Operator Runbook

This is the one-file guide for driving the DroneCaptureOps Gym project to a complete OpenEnv India Hackathon 2026 submission. It is written so you can hand it to a chatbot, teammate, or future you and walk through it step by step.

## Deadline And Submission Rules (READ FIRST)

- Deadline: 26 April 2026, 17:00 IST. Hard cutoff. Changes or commits after the deadline are not considered.
- Only ONE submission per team. Pick the strongest single idea.
- Only the Team Leader can submit. Use the email registered for the team. For this team that means submitting from `gootysaivishwas12@gmail.com`.
- Every URL you submit must be publicly accessible. Open every URL in an Incognito/Private browser window before pasting it into the form.
- Do not include large video files in the HF environment repo. Reference videos via public URLs only.

If a section in this runbook does not directly affect one of the four submission form fields, it is not blocking the submission. Do that work only after the form is submitted.

## Submission Form Fields (Authoritative)

You must submit exactly these four URLs in the official Google Form:

1. Hugging Face Space URL for your environment.
   - Must be a public Space.
   - Must serve the OpenEnv FastAPI server (`server.app:app`) so judges can pull and evaluate.
2. Training Run Notebook URL.
   - Public Google Colab link, OR
   - HF Space repository URL containing the `.ipynb`, OR
   - Public Kaggle Notebook URL.
3. YouTube Demo Video URL OR Blog Post URL (pick one in the form).
4. The actual URL for whichever option you picked above.
   - YouTube: a public YouTube URL.
   - Blog Post: the URL of a Markdown blog file inside your Hugging Face environment repository (not an external blog).

Anything else you do this week is supporting work for those four URLs.

## Low-Effort Fallback Plan (If You Are Short On Time)

If you cannot record a polished YouTube video before the deadline, take the blog-post path. The form explicitly allows a blog post in place of a YouTube video, but it must be a Markdown file living inside your Hugging Face environment repository.

Minimum viable submission package, in order:

1. Push the repo to a public Hugging Face Space so the OpenEnv FastAPI server runs there.
2. Add a `notebooks/training_run.ipynb` (or similarly named) executed notebook to the same Space repo. The training notebook URL becomes the file URL on Hugging Face.
3. Add a `BLOG.md` (or similar) Markdown blog file to the same Space repo. The blog post URL becomes the file URL on Hugging Face. (`BLOG.md` is already present at the repo root and will be pushed to the Space.)
4. Update the README in the Space to link the notebook and blog file.
5. Submit those URLs in the form.

This way you only need one public Hugging Face Space and you do not need YouTube, Colab, Kaggle, or external blog hosting.

## Scope Of This Runbook

- Keep the repo submission-ready while teammates continue training experiments.
- Validate the OpenEnv environment locally and deploy it to a public Hugging Face Space.
- Get a Training Run Notebook public, even if minimal.
- Get either a public YouTube video or a Markdown blog file inside the HF Space repo.
- Make the README link the HF Space, the notebook, and the video/blog.
- Avoid overclaiming anything that is not implemented or trained yet.

## 0. Current Truth

Before doing anything, keep this mental model:

- DroneCaptureOps is an OpenEnv-compatible RL environment for LLM drone inspection directors.
- The submitted environment currently trains/evaluates on the deterministic `GeometryController`.
- `DroneKitSITLController` exists as a placeholder boundary, not a finished backend.
- DroneKit footage should be presented as a bridge/SITL demo unless someone fully implements and validates the backend.
- Real training plots/results must come from real runs. Do not fabricate evidence.
- Large video files should not be committed. Upload video externally and link it from README.

The strongest story:

> DroneCaptureOps teaches LLMs to gather evidence before answering. The agent directs high-level drone inspection tools, collects RGB/thermal evidence, avoids unsafe actions, and submits a grounded report tied to real photo IDs.

## 0.1 Completed Local Checks

These checks have already been run from the repo root. Do not repeat them unless code changes again before submission.

Completed:

- Local machine setup verified:
  - `python3 --version`: `Python 3.9.6` (system default; do not use this for repo validation).
  - `python3.11 --version`: `Python 3.11.14` (use this for repo validation).
  - `git --version`: `git version 2.39.5 (Apple Git-154)`.
  - `docker --version`: `Docker version 28.1.1, build 4eba377`.
- Repo navigation verified:
  - `pwd`: `/Users/saivishwasgooty/Documents/DroneCaptureGym`.
  - Required repo files/directories are present: `README.md`, `pyproject.toml`, `openenv.yaml`, `server/`, `dronecaptureops/`, and `inference.py`.
- Submission blog fallback prepared:
  - `BLOG.md` exists at the repo root and should be pushed to the Hugging Face Space if using the Blog Post form option.
- `pytest` passed: `334 passed in 61.49s`.
- `python3.11 inference.py --policy scripted` passed:
  - `basic_thermal_survey`: success, `score=1.00`.
  - `anomaly_confirmation`: success, `score=1.00`.
  - `audit_grade_strict_grounding`: success, `score=1.00`.
- `openenv validate` passed with `[OK] DroneCaptureGym: Ready for multi-mode deployment`.
- Local OpenEnv `/reset` was previously checked against `http://127.0.0.1:8000/reset` and returned HTTP 200.

Still blocked:

- Docker is installed, but Docker Desktop/daemon is not running.
- `docker info` fails with: `Cannot connect to the Docker daemon at unix:///Users/saivishwasgooty/.docker/run/docker.sock. Is the docker daemon running?`
- The full validator reached Step 2 and stopped because Docker was unavailable.

Next action from this section:

1. Start Docker Desktop.
2. Run `docker info`.
3. Run `docker build -t dronecaptureops-gym .`.
4. After the real HF Space URL exists, run `./scripts/validate-submission.sh <REAL_HF_SPACE_URL> .`.

## 1. Prerequisites

### Accounts

You need:

- GitHub account or wherever the public repo will live.
- Hugging Face account.
- Permission to create a public Hugging Face Space.
- Permission to upload/link a public video, blog post, or slide deck.
- Optional: Hugging Face token with repo write permissions if pushing artifacts/models.

### Local Machine

Recommended:

- Python 3.11+ (`python3.11 --version` verified: `Python 3.11.14`; system `python3` is only `Python 3.9.6`, so use `python3.11` for validation commands).
- Git (verified: `git version 2.39.5 (Apple Git-154)`).
- Docker Desktop / Docker CLI (CLI verified: `Docker version 28.1.1, build 4eba377`; daemon is still blocked until Docker Desktop is started).
- A terminal with the repo checked out (verified at `/Users/saivishwasgooty/Documents/DroneCaptureGym`).
- Browser
- Screen recorder
- Optional ground station / map viewer for DroneKit SITL:
  - Mission Planner
  - QGroundControl
  - MAVProxy map/console

### Repo Location

The working repo is:

```bash
/Users/saivishwasgooty/Documents/DroneCaptureGym
```

Use this as the working directory:

```bash
cd /Users/saivishwasgooty/Documents/DroneCaptureGym
```

Already verified:

```bash
pwd
# /Users/saivishwasgooty/Documents/DroneCaptureGym

ls
# Required files/directories present:
# README.md
# pyproject.toml
# openenv.yaml
# server/
# dronecaptureops/
# inference.py
```

## 2. Submission Owner Job

Your role is not to keep fixing every failing RL experiment. Your role is to keep one branch clean and submission-ready.

Do this:

1. Create or identify a `submission-ready` branch.
2. Only merge changes that improve the final submission package.
3. Keep teammate training experiments on separate branches.
4. Protect the story: do not let the README claim results before results exist.
5. Protect the release gate: tests, OpenEnv validation, Docker build, and Space validation must pass before final submission.

Recommended release gate:

```bash
pytest
python inference.py --policy scripted
openenv validate
docker build -t dronecaptureops-gym .
./scripts/validate-submission.sh <REAL_HF_SPACE_URL> .
```

## 3. Local Repo Validation

### 3.1 Set Up Python

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If training tools are needed later:

```bash
pip install -e ".[dev,train]"
```

### 3.2 Run Tests

```bash
pytest
```

Expected:

- Tests pass.
- Hidden-state leakage tests pass.
- Inference CLI tests pass.
- Reward tests pass.

If this fails:

- Do not deploy yet.
- Save the failure output.
- Fix only submission-blocking issues on the stable branch.
- Keep unrelated training experiments isolated.

### 3.3 Run Inference Baseline

```bash
python inference.py --policy scripted
```

Expected:

- It runs three deterministic tasks.
- Output includes `[START]`, `[STEP]`, and `[END]`.
- It should not crash.

Example shape:

```text
[START] task=basic_thermal_survey env=dronecaptureops-gym model=scripted
[STEP] step=1 action=... reward=0.00 done=false error=null
[END] success=true steps=24 score=1.00 rewards=...
```

### 3.4 Validate OpenEnv

```bash
openenv validate
```

Expected:

- OpenEnv recognizes `openenv.yaml`.
- The app target `server.app:app` is valid.
- Validation passes.

### 3.5 Run Local Server

Start the server:

```bash
dronecaptureops-server
```

or:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
curl -X POST http://127.0.0.1:8000/reset -H "Content-Type: application/json" -d '{}'
```

Expected:

- HTTP 200.
- JSON observation response.

## 4. Docker Validation

Start Docker Desktop first.

Then:

```bash
docker build -t dronecaptureops-gym .
```

If the build succeeds, run it:

```bash
docker run --rm -p 8000:8000 dronecaptureops-gym
```

In another terminal:

```bash
curl -X POST http://127.0.0.1:8000/reset -H "Content-Type: application/json" -d '{}'
```

Expected:

- Container starts.
- `/reset` returns HTTP 200.

If Docker fails:

- Confirm Docker Desktop is running.
- Confirm the repo root has `Dockerfile`.
- Confirm `pyproject.toml`, `README.md`, `openenv.yaml`, `dronecaptureops/`, `server/`, and `inference.py` are copied by the Dockerfile.

## 5. Hugging Face Space Deployment

### 5.1 Create Space

On Hugging Face:

1. Create a new public Space.
2. Use a clear name, for example `dronecaptureops-gym`.
3. Choose a Docker or compatible deployment path if needed.
4. Push the repo contents required to run the OpenEnv server.

The Space must serve:

```text
server.app:app
```

The repo already has:

```yaml
spec_version: 1
name: dronecaptureops-gym
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### 5.2 Check Space Manually

After the Space is live, check:

```bash
curl -X POST <REAL_HF_SPACE_URL>/reset -H "Content-Type: application/json" -d '{}'
```

Expected:

- HTTP 200.
- Observation JSON.

### 5.3 Run Full Submission Validator

```bash
./scripts/validate-submission.sh <REAL_HF_SPACE_URL> .
```

Expected:

- Step 1: HF Space responds to `/reset`.
- Step 2: Docker build succeeds.
- Step 3: `openenv validate` passes.

Do not submit until this passes.

## 6. README Finalization

The README currently contains placeholders. That is good while links are not final, but not acceptable at submission time.

Before final submission, search for placeholders:

```bash
rg "TODO_|PLACEHOLDER|<REAL_HF_SPACE_URL>" README.md docs
```

Replace all public submission placeholders:

- `TODO_HF_SPACE_URL`
- `TODO_RESULTS_SUMMARY`
- `TODO_LOSS_PLOT_URL`
- `TODO_REWARD_PLOT_URL`
- `TODO_EXPERIMENT_TRACKING_URL`
- `TODO_EXECUTED_TRAINING_NOTEBOOK_URL`
- `TODO_TRAINED_MODEL_OR_ADAPTER_URL`
- `TODO_WRITEUP_OR_VIDEO_URL`
- `TODO_OPTIONAL_PRESENTATION_URL`

Minimum final README links:

- Public Hugging Face Space.
- Public video/blog/slides.
- Real training/eval evidence from teammates.
- Real plots or tracker link.
- Model/artifact link if available.

Do not link private/local files.

## 7. License (Post-Deadline / Optional)

The submission form does not ask for a license. Adding one is good hygiene for a public repo, but it is not required for submission.

Do this only after the form is submitted:

1. Ask the team which license they want.
2. Common options:
   - MIT: simple and permissive.
   - Apache-2.0: permissive with explicit patent language.
   - BSD-3-Clause: permissive and conservative.
3. Add a root `LICENSE`.

Do not delay submission to wait on a license decision.

## 8. DroneKit Demo Setup

This is for the video/demo story. It is not required for the OpenEnv Space unless a real adapter is implemented.

### 8.1 What You Are Showing

Safe, honest message:

> The submitted RL environment is simulator-first for speed and determinism. The command layer is backend-neutral, and here we show how high-level inspection actions map to DroneKit/ArduPilot SITL.

Do not say:

> The submitted Space directly controls a real drone.

unless that is fully implemented, tested, and safe.

### 8.2 Install DroneKit Demo Environment

Use a separate venv so old DroneKit dependencies do not pollute the project environment:

```bash
python -m venv .venv-dronekit-demo
source .venv-dronekit-demo/bin/activate
pip install dronekit dronekit-sitl MAVProxy
```

If install fails on modern Python, try Python 3.10 or 3.9 for the demo venv. DroneKit is old and may be sensitive to dependency versions.

### 8.3 Start ArduPilot SITL

Terminal 1:

```bash
source .venv-dronekit-demo/bin/activate
dronekit-sitl copter
```

Expected:

- SITL downloads/starts a copter binary.
- It waits for TCP connections on `127.0.0.1:5760`.

### 8.4 Optional: Add MAVProxy For Map/GCS

Terminal 2:

```bash
source .venv-dronekit-demo/bin/activate
mavproxy.py --master tcp:127.0.0.1:5760 --out 127.0.0.1:14550 --out 127.0.0.1:14551
```

Then:

- DroneKit script connects to `127.0.0.1:14550`.
- Mission Planner/QGroundControl connects to `127.0.0.1:14551`.

If you skip MAVProxy, a simple script can connect directly to:

```text
tcp:127.0.0.1:5760
```

### 8.5 Minimal DroneKit Demo Script

Create a temporary local file outside the committed repo if this is only for shooting:

```python
import time
from dronekit import LocationGlobalRelative, VehicleMode, connect


CONNECTION = "127.0.0.1:14550"  # or "tcp:127.0.0.1:5760" without MAVProxy

vehicle = connect(CONNECTION, wait_ready=True)

print("Waiting for vehicle to become armable")
while not vehicle.is_armable:
    time.sleep(1)

print("Switching to GUIDED and arming")
vehicle.mode = VehicleMode("GUIDED")
vehicle.armed = True

while not vehicle.armed:
    time.sleep(1)

target_altitude = 10
print(f"Taking off to {target_altitude}m")
vehicle.simple_takeoff(target_altitude)

while vehicle.location.global_relative_frame.alt < target_altitude * 0.95:
    print("Altitude:", vehicle.location.global_relative_frame.alt)
    time.sleep(1)

vehicle.airspeed = 3

print("OpenEnv tool: fly_to_viewpoint -> DroneKit: simple_goto leg 1")
vehicle.simple_goto(LocationGlobalRelative(-35.361354, 149.165218, 20), groundspeed=5)
time.sleep(15)

print("OpenEnv tool: fly_to_viewpoint -> DroneKit: simple_goto leg 2")
vehicle.simple_goto(LocationGlobalRelative(-35.363244, 149.168801, 20), groundspeed=5)
time.sleep(15)

print("OpenEnv tool: return_home -> DroneKit: RTL")
vehicle.mode = VehicleMode("RTL")
time.sleep(5)

vehicle.close()
```

Run it:

```bash
python dronekit_demo.py
```

Expected:

- Console prints arm/takeoff/goto/RTL.
- Map/GCS shows simulated drone moving.

### 8.6 OpenEnv-To-DroneKit Mapping

Use this mapping in narration:

| OpenEnv Tool | DroneKit/SITL Equivalent |
| --- | --- |
| `takeoff` | `VehicleMode("GUIDED")`, `vehicle.armed = True`, `vehicle.simple_takeoff(altitude_m)` |
| `fly_to_viewpoint` | `vehicle.simple_goto(LocationGlobalRelative(...), groundspeed=speed_mps)` |
| `return_home` | `vehicle.mode = VehicleMode("RTL")` |
| `land` | `vehicle.mode = VehicleMode("LAND")` |
| `set_gimbal` | Future MAVLink/gimbal adapter work |
| `capture_rgb` / `capture_thermal` | Simulated capture model today; real payload adapter later |
| `submit_evidence_pack` | OpenEnv/reporting layer, not DroneKit motor control |

## 9. Physical Drone Footage

If you include a real drone in the video, treat it as a safety-critical shoot.

Recommended:

- Use the physical drone for B-roll, takeoff/hover footage, or manual demonstration.
- Use DroneKit/SITL for actual LLM-to-autopilot command demonstration.
- Keep a manual pilot in control.
- Fly only in a legal, safe, authorized location.
- Avoid people, roads, animals, buildings, and tight indoor spaces.
- Do not run untested LLM outputs directly on real hardware.

Good narration:

> For safety, the RL benchmark runs in a deterministic simulator. The same high-level command layer can be bridged to DroneKit, and here we show that bridge in ArduPilot SITL.

## 10. Video Production Plan

The final video should be 90-120 seconds and include three people.

### Roles

Person A: story lead.

- Opens the video.
- Explains why this is not a toy benchmark.
- Closes with the impact.

Person B: environment/RL lead.

- Explains OpenEnv.
- Explains observations, tools, rewards, and training evidence.
- Shows terminal/HF Space/README.

Person C: DroneKit/demo lead.

- Shows SITL/DroneKit/physical drone footage.
- Explains how high-level commands map to DroneKit.
- Handles safety language.

### Shot List

Record these:

1. Three people on camera with drone visible.
2. Close-up of drone or controller.
3. README or HF Space page.
4. Terminal running:

```bash
python inference.py --policy scripted
```

5. OpenEnv `[START]`, `[STEP]`, `[END]` output.
6. Reward formula or reward breakdown in README/trace.
7. DroneKit SITL/map view moving after `simple_goto`.
8. Real training/eval plot when available.
9. Final README links section.

### Split-Screen Moment

Best 10-second visual:

Left side:

```text
OpenEnv tool: fly_to_viewpoint
```

Right side:

```text
DroneKit command: vehicle.simple_goto(...)
```

Map:

```text
Simulated drone moves to inspection waypoint
```

## 11. Three-Person Video Script

### 0:00-0:10 - Hook

Visual: three people on camera, drone visible.

Person A:

> What if making an LLM operate a drone was not a hard-coded demo, but a trainable RL environment?

Person B:

> That is DroneCaptureOps Gym: an OpenEnv benchmark where the LLM becomes the inspection director.

Person C:

> Not raw motor control. High-level drone operations: inspect, capture evidence, stay safe, and submit a grounded report.

### 0:10-0:28 - Problem

Visual: solar farm map, environment trace, or drone B-roll.

Person A:

> Real drone inspection is not just flying from point A to point B. The hard part is knowing what evidence is missing.

Person A:

> Did we capture every required solar row? Is the thermal photo good enough? Did we confirm the anomaly with RGB? Can the final report cite real photo IDs?

### 0:28-0:48 - Environment

Visual: terminal or HF Space showing an OpenEnv run.

Person B:

> The agent sees mission state, telemetry, a site map, visible assets, capture logs, and a dynamic tool catalog.

Person B:

> It acts through structured OpenEnv tools: take off, fly to a viewpoint, point the camera, capture thermal or RGB evidence, inspect a photo, return home, land, and submit the evidence pack.

### 0:48-1:08 - DroneKit Demo

Visual: split screen with OpenEnv action on left and DroneKit SITL/map on right.

Person C:

> The training environment is simulator-first so it is fast and reproducible. But the command layer is designed like a real drone stack.

Person C:

> Here, a high-level inspection action maps into DroneKit: GUIDED mode, arm, takeoff, simple-goto, and return-to-launch in ArduPilot SITL.

On-screen text:

```text
OpenEnv tool: fly_to_viewpoint
DroneKit command: vehicle.simple_goto(...)
```

### 1:08-1:30 - Reward

Visual: reward formula or reward breakdown.

Person B:

> The reward is not a single pass/fail score. It teaches the process: coverage, capture quality, issue confirmation, battery management, safety, and report grounding.

Person A:

> And it is hard to game. If the agent violates safety, the score is capped. If it cites fake photos or unsupported findings, the integrity gate caps the score.

### 1:30-1:48 - Results

Visual: real plot/tracker when available.

Person B:

> We compare trained behavior against random and baseline policies across deterministic inspection tasks.

Person B:

> The final README links the real training run, reward curves, loss curves, and model artifacts so judges can verify the result.

Replace after training:

> After training, the model improved from `TODO_BASELINE_SCORE` to `TODO_TRAINED_SCORE` on `TODO_EVAL_SUITE`, with better grounded evidence collection and fewer unsupported report claims.

### 1:48-2:00 - Close

Visual: HF Space and README links.

Person C:

> DroneCaptureOps is about making LLMs useful in the field: gather evidence first, then answer.

Person A:

> The OpenEnv Space, code, validation commands, video, and results are all linked from the README.

## 12. Final Submission Checklist

Confirm the four form fields, then engineering hygiene.

### 12.1 Form-Field Readiness (Blocking)

- Hugging Face Space URL is public, opens in Incognito, and `/reset` returns HTTP 200.
- Training Run Notebook URL is public:
  - Colab share link set to "Anyone with the link can view", or
  - HF Space repo file URL pointing at the executed `.ipynb`, or
  - Public Kaggle Notebook URL.
- One of:
  - Public YouTube video URL (Unlisted with a public link is acceptable; do not use Private), or
  - Markdown blog file URL inside the Hugging Face environment repository.
- README in the HF Space links the HF Space itself, the notebook, and the video/blog.

### 12.2 Engineering Hygiene (Strong Recommended)

- `pytest` passes locally.
- `python inference.py --policy scripted` runs locally.
- `openenv validate` passes.
- Docker build passes locally.
- `./scripts/validate-submission.sh <REAL_HF_SPACE_URL> .` passes.
- README has no unresolved public-link placeholders.
- README has real (or honestly described) training/eval evidence.
- No large video files are committed to the HF Space repo.
- DroneKit claims in README/video are honest and match what is implemented.

### 12.3 Incognito Verification (Do This Last)

Open a fresh Incognito/Private window, sign out of Hugging Face/Google/YouTube/Kaggle, and confirm that each URL loads and the content is visible. If anything 404s or asks for login, the submission will count as broken.

Verify:

- HF Space URL.
- Training Run Notebook URL.
- YouTube video URL or blog Markdown file URL.
- The Hugging Face Space README (it must contain the same links).

## 13. What To Ask A Chatbot To Do With This File

Prompt:

```text
You are helping me prepare the DroneCaptureOps Gym OpenEnv hackathon submission.
Use this runbook as the source of truth.
Walk me through the steps one by one.
At each step, tell me exactly what command to run or what artifact to collect.
Do not let me claim training results, DroneKit support, or deployment status unless we have evidence.
Start from Section 1 and wait for my confirmation after each major section.
```

If blocked, ask:

```text
I am blocked at section <number>. Here is the output/error: <paste output>.
Diagnose only this step and tell me the next safest action.
```

## 14. Immediate Next Actions (Deadline-Aware)

Today is the deadline. Treat this as a triage list. Get the four form URLs first, polish second.

### 14.1 Critical Path To A Valid Submission

Do these in order. Each one unblocks the next.

1. Local repo health is already checked unless code changes again:
   - `pytest` passed.
   - `python3.11 inference.py --policy scripted` passed.
   - `openenv validate` passed.
2. Start Docker Desktop and run `docker build -t dronecaptureops-gym .` if there is time. If Docker still blocks, do not let this delay the four required form URLs.
3. Push the current repo to a public Hugging Face Space and confirm `/reset` returns HTTP 200 in Incognito.
4. Get the Training Run Notebook public:
   - Easiest path: commit the executed `.ipynb` directly into the HF Space repo and use that file URL.
   - Alternate paths: Colab share link or public Kaggle notebook.
5. Choose the YouTube video OR blog post path:
   - YouTube path: record, upload as Public or Unlisted, copy the URL.
   - Blog path (already prepared): `BLOG.md` exists at the repo root. After pushing the repo to the HF Space, the public blog URL will be `https://huggingface.co/spaces/<owner>/<space-name>/blob/main/BLOG.md`. Open this URL once in Incognito to confirm it loads, then paste it into the form.
6. Update the README in the HF Space repo so it links:
   - The HF Space URL itself.
   - The notebook URL.
   - The YouTube or blog URL.
7. Open every URL in an Incognito window and confirm they all load.
8. Team Leader (`gootysaivishwas12@gmail.com`) submits the four URLs in the official form.

### 14.2 If You Have Time After The Critical Path

Only after step 8 above:

- Run the full validator: `./scripts/validate-submission.sh <REAL_HF_SPACE_URL> .`
- Record extra B-roll for a polished video.
- Set up DroneKit SITL and record the bridge demo as bonus media.
- Add a `LICENSE` file.
- Replace any remaining `TODO_*` placeholders with real links.
- Add real training plots to the README and Space.

### 14.3 What Not To Do Today

- Do not start a long RL training run from scratch and block on it. Use whatever real evidence already exists.
- Do not try to make `DroneKitSITLController` a real backend on submission day.
- Do not commit large video files to the HF Space repo.
- Do not change anything after the deadline. The form ignores post-deadline edits.
