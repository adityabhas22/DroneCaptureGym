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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run Tests

```bash
pytest
```

## Run One Episode

```bash
python examples/run_scripted_agent.py
python examples/run_random_agent.py
```

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

- Mission/map: `get_site_map`, `get_mission_checklist`, `get_telemetry`, `estimate_view`
- Flight: `takeoff`, `fly_to_viewpoint`, `hover`, `return_home`, `land`
- Camera: `set_gimbal`, `set_zoom`, `capture_rgb`, `capture_thermal`, `inspect_capture`
- Evidence: `mark_target_inspected`, `submit_evidence_pack`

## Reward Shape

Rewards are logged as independent components:

- `format_validity`
- `flight_success`
- `target_coverage`
- `capture_quality`
- `defect_visibility`
- `checklist_completion`
- `route_efficiency`
- `battery_management`
- `safety_compliance`
- `report_grounding`
- `recovery_behavior`

The total reward uses a safety gate so serious safety violations cap the episode value. The environment rewards grounded inspection evidence, not just reaching waypoints.

## Hidden State

Hidden defects and verifier labels live only in internal `EpisodeWorld`. Observations and visible state expose mission, map, telemetry, captures, reward breakdown, and checklist status, but not the hidden expected solution.

## Demo Direction

A weak baseline should fly to a generic overview point, capture one or two low-value photos, miss an anomaly or close-up, and submit an incomplete pack. A stronger trained agent should read the checklist, capture thermal overview, inspect quality, reposition for missing rows or anomalies, capture RGB close-up evidence, avoid the no-fly zone, return home, land, and submit a photo-linked evidence pack.
