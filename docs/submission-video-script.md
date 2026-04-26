# DroneCaptureOps Submission Video Script

Target length: 90-120 seconds for the submitted cut. Record more raw footage than you need, then cut aggressively.

Purpose: make judges instantly understand the ambition: DroneCaptureOps is an OpenEnv RL environment for turning an LLM into a high-level drone inspection director. The video should feel concrete, not theoretical: show the environment, show tool calls, show DroneKit/SITL or a real drone safety demo, show the reward story, and point to the final README links.

Do not commit the final video to the repo. Upload it externally and replace `TODO_WRITEUP_OR_VIDEO_URL` in the README.

## Eye-Catching Pitch

Use this as the first line:

> What if training an LLM to operate a drone was as simple as running it through an RL environment?

Then immediately clarify the safety boundary:

> DroneCaptureOps does not teach raw motor control. It teaches the decision layer: where to inspect, what evidence is missing, when to recapture, and how to submit a grounded inspection report.

## Three-Person Roles

Person A: product/story lead.

- Opens with the hook.
- Explains the real-world problem.
- Keeps the video understandable for non-drone viewers.

Person B: environment/RL lead.

- Explains OpenEnv, the action space, observations, rewards, and what the agent learns.
- Shows `inference.py`, reward breakdowns, traces, or HF Space.

Person C: DroneKit/demo lead.

- Shows the drone/SITL demo.
- Explains how high-level agent decisions map to DroneKit commands.
- Owns safety language: simulation first, physical drone only under controlled/manual conditions.

## Important Honesty Note

The repository's production training path currently uses `GeometryController`, which is deterministic and fast for RL. The `DroneKitSITLController` file exists as a placeholder boundary, not a finished backend.

For the video, there are two honest ways to show DroneKit:

1. **Recommended:** show the LLM/OpenEnv policy producing high-level drone actions, then show a separate DroneKit/SITL bridge executing the same kind of high-level commands in ArduPilot SITL. Narration: "The environment is backend-neutral; here is the same command layer driving DroneKit/SITL."
2. **Only if implemented and tested:** wire `DroneKitSITLController` fully into `DroneCaptureOpsEnvironment` and show OpenEnv directly controlling SITL.

Do not claim the submitted OpenEnv Space controls a real drone unless that path is implemented, tested, and safe.

## DroneKit Research Summary

DroneKit-Python connects to an ArduPilot vehicle or simulator with `connect(connection_string, wait_ready=True)`. Common connection strings include `tcp:127.0.0.1:5760` for DroneKit-SITL and `127.0.0.1:14550` for UDP forwarding through MAVProxy.

For Copter, a standard guided demo flow is:

1. Connect to vehicle.
2. Wait until `vehicle.is_armable`.
3. Set `vehicle.mode = VehicleMode("GUIDED")`.
4. Set `vehicle.armed = True`.
5. Wait until armed.
6. Call `vehicle.simple_takeoff(target_altitude)`.
7. Monitor `vehicle.location.global_relative_frame.alt` because takeoff is asynchronous.
8. Send `vehicle.simple_goto(LocationGlobalRelative(...), groundspeed=...)`.
9. Return with `vehicle.mode = VehicleMode("RTL")` or land with `VehicleMode("LAND")`.
10. Close the vehicle connection.

Useful references:

- [DroneKit connecting to a vehicle](https://dronekit-python.readthedocs.io/en/latest/guide/connecting_vehicle.html)
- [DroneKit simple goto example](https://dronekit-python.readthedocs.io/en/latest/examples/simple_goto.html)
- [DroneKit SITL setup](https://dronekit-python.readthedocs.io/en/latest/develop/sitl_setup.html)
- [DroneKit guided movement example](https://dronekit-python.readthedocs.io/en/latest/examples/guided-set-speed-yaw-demo.html)

## Recommended DroneKit Shoot Plan

### Demo Goal

Show a believable chain:

```text
LLM policy -> OpenEnv tool call -> high-level inspection action -> DroneKit/SITL command -> drone moves on map
```

The judge should walk away thinking: "This is not another toy grid world. They built the training layer for LLM drone operations, and it can map to a real autopilot stack."

### Safest Version To Shoot

Use ArduPilot SITL plus a map/GCS view. If you include a physical drone, use it as a visual prop or a manually supervised demo, not as unsupervised autonomous flight.

Screen layout for recording:

- Left: terminal running `python inference.py --policy scripted` or model-backed inference.
- Middle: a small bridge terminal printing mapped DroneKit commands, for example `takeoff -> simple_takeoff(10)`.
- Right: Mission Planner, QGroundControl, MAVProxy map, or another visualizer showing the simulated drone moving.

### Physical Drone Footage

If you operate a real drone:

- Keep it outdoors or in a safe authorized flight area.
- Have a manual pilot ready to override.
- Avoid crowds, roads, animals, reflective surfaces, and indoor prop hazards.
- Do not show reckless autonomy.
- Consider filming the physical drone taking off/hovering manually, while the real "LLM control" demo happens in SITL on screen.

Best honest line:

> For safety, the submission runs the inspection benchmark in a deterministic OpenEnv simulator. The same high-level command layer can be bridged to DroneKit, and here we show that bridge in ArduPilot SITL.

## DroneKit Demo Setup

These commands are for a local shoot machine. Keep this outside the final HF Space unless you actually implement the backend.

Install demo dependencies in a separate environment:

```bash
python -m venv .venv-dronekit-demo
source .venv-dronekit-demo/bin/activate
pip install dronekit dronekit-sitl MAVProxy
```

Start SITL:

```bash
dronekit-sitl copter
```

Expected behavior: SITL waits for TCP connections on `127.0.0.1:5760`.

Optional: forward SITL to separate UDP ports so DroneKit and a ground station can both connect:

```bash
mavproxy.py --master tcp:127.0.0.1:5760 --out 127.0.0.1:14550 --out 127.0.0.1:14551
```

Then:

- Connect DroneKit script to `127.0.0.1:14550`.
- Connect Mission Planner/QGroundControl/map viewer to `127.0.0.1:14551`.

Minimal DroneKit movement script for the demo:

```python
import time
from dronekit import LocationGlobalRelative, VehicleMode, connect


vehicle = connect("127.0.0.1:14550", wait_ready=True)

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
print("Flying inspection leg 1")
vehicle.simple_goto(LocationGlobalRelative(-35.361354, 149.165218, 20), groundspeed=5)
time.sleep(15)

print("Flying inspection leg 2")
vehicle.simple_goto(LocationGlobalRelative(-35.363244, 149.168801, 20), groundspeed=5)
time.sleep(15)

print("Returning to launch")
vehicle.mode = VehicleMode("RTL")
time.sleep(5)

vehicle.close()
```

## Mapping OpenEnv Tools To DroneKit Shots

Use these mappings in the demo narration:

- `takeoff` maps to `VehicleMode("GUIDED")`, `vehicle.armed = True`, and `vehicle.simple_takeoff(altitude_m)`.
- `fly_to_viewpoint` maps to `vehicle.simple_goto(LocationGlobalRelative(...), groundspeed=speed_mps)`.
- `return_home` maps to `vehicle.mode = VehicleMode("RTL")`.
- `land` maps to `vehicle.mode = VehicleMode("LAND")`.
- `set_gimbal`, `capture_rgb`, and `capture_thermal` are modeled in the OpenEnv simulator today; real payload support would be the next DroneKit/MAVLink adapter step.

Do not overpromise camera hardware control unless you have it working.

## What To Film

Shot list:

1. Three-person intro, 5 seconds.
2. Close-up of drone or drone controller, 3 seconds.
3. README/HF Space page with project title, 5 seconds.
4. Terminal showing `[START]`, `[STEP]`, `[END]` from `inference.py`, 8 seconds.
5. OpenEnv action catalog or reward breakdown, 8 seconds.
6. DroneKit SITL/map view moving after a command, 15 seconds.
7. One training/result visual when real evidence exists, 8 seconds.
8. Final README links section, 5 seconds.

## 90-120 Second Three-Person Script

### 0:00-0:10 - Hook

Visual: three people on camera, drone visible on table or in background.

Person A:

> What if making an LLM operate a drone was not a hard-coded demo, but a trainable RL environment?

Person B:

> That is DroneCaptureOps Gym: an OpenEnv benchmark where the LLM becomes the inspection director.

Person C:

> Not raw motor control. High-level drone operations: inspect, capture evidence, stay safe, and submit a grounded report.

### 0:10-0:28 - Problem

Visual: solar farm map or environment trace.

Person A:

> Real drone inspection is not just flying from point A to point B. The hard part is knowing what evidence is missing.

Person A:

> Did we capture every required solar row? Is the thermal photo good enough? Did we confirm the anomaly with RGB? Can the final report cite real photo IDs?

### 0:28-0:48 - Environment

Visual: terminal or HF Space showing OpenEnv run.

Person B:

> The agent sees mission state, telemetry, a site map, visible assets, capture logs, and a dynamic tool catalog.

Person B:

> It acts through structured OpenEnv tools: take off, fly to a viewpoint, point the camera, capture thermal or RGB evidence, inspect a photo, return home, land, and submit the evidence pack.

### 0:48-1:08 - DroneKit Demo

Visual: split screen with OpenEnv/LLM action on the left and DroneKit SITL or drone footage on the right.

Person C:

> The training environment is simulator-first so it is fast and reproducible. But the command layer is designed like a real drone stack.

Person C:

> Here, a high-level inspection action maps into DroneKit: GUIDED mode, arm, takeoff, `simple_goto`, and return-to-launch in ArduPilot SITL.

On-screen text:

```text
OpenEnv tool: fly_to_viewpoint
DroneKit command: vehicle.simple_goto(...)
```

### 1:08-1:30 - Reward

Visual: reward formula or reward breakdown from README/trace.

Person B:

> The reward is not a single pass/fail score. It teaches the process: coverage, capture quality, issue confirmation, battery management, safety, and report grounding.

Person A:

> And it is hard to game. If the agent violates safety, the score is capped. If it cites fake photos or unsupported findings, the integrity gate caps the score.

### 1:30-1:48 - Results Placeholder

Visual: real plot/tracker when available. Until then, use a slide that says "Real training evidence linked in README" only for internal rehearsal, not final submission.

Person B:

> We compare trained behavior against random and baseline policies across deterministic inspection tasks.

Person B:

> The final README links the real training run, reward curves, loss curves, and model artifacts so judges can verify the result.

Replace with exact result after training:

> After training, the model improved from `TODO_BASELINE_SCORE` to `TODO_TRAINED_SCORE` on `TODO_EVAL_SUITE`, with better grounded evidence collection and fewer unsupported report claims.

### 1:48-2:00 - Close

Visual: HF Space and README links.

Person C:

> DroneCaptureOps is about making LLMs useful in the field: gather evidence first, then answer.

Person A:

> The OpenEnv Space, code, validation commands, video, and results are all linked from the README.

## Backup 60-Second Cut

Person A:

> Most LLM benchmarks test answers. DroneCaptureOps tests whether an LLM can gather the evidence needed to answer.

Person B:

> The agent operates as a drone inspection director through OpenEnv tools: plan the mission, fly safe viewpoints, capture RGB and thermal evidence, inspect quality, and submit a photo-grounded report.

Person C:

> We also show how those high-level actions map to DroneKit and ArduPilot SITL: takeoff, guided movement, and return-to-launch.

Person B:

> The reward teaches process and outcome: coverage, issue capture, safety, efficiency, and grounded reporting. Unsafe shortcuts and fake citations cap the score.

Person A:

> The goal is simple: make LLMs better at active, safety-aware inspection work, not just text prediction.

## Final Recording Checklist

- Keep the final cut under 2 minutes.
- Show one short `python inference.py --policy scripted` run.
- Show the HF Space or local OpenEnv server if the public Space is not live yet.
- Show DroneKit in SITL or a controlled safety demo.
- Do not imply physical autonomous flight unless it is actually safe, legal, and implemented.
- Show real training/eval plots only after they exist.
- Upload externally and replace `TODO_WRITEUP_OR_VIDEO_URL` in the README.
