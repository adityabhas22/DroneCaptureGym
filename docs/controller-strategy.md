# Controller Strategy

The `DroneController` boundary in `dronecaptureops/controllers/base.py` lets us swap simulator backends without touching the env loop, tools, or rewards. The MVP commits to a deterministic geometry simulator. SITL/ArduPilot is an explicit non-goal until the benchmark surface stabilizes.

## Backends

| Backend | File | Status | When to use |
|---|---|---|---|
| `GeometryController` | `controllers/geometry_controller.py` | **Default**. Deterministic, CPU-light, no rendering. Used by every test, suite, and trace. | All MVP work. |
| `DroneKitSITLController` | `controllers/dronekit_sitl_controller.py` | **Placeholder**. Every controller method raises `NotImplementedError`. The class exists only to hold the boundary stable. | Not yet. |

## Why we are NOT investing in SITL right now

- Reward, evidence, and integrity gates are still moving. SITL adds ~1–10s of episode wall time and a non-Python dependency tree (ArduPilot + MAVLink). Iterating reward math against that is too slow.
- The `DroneController` interface is small (takeoff, fly_to, hover, set_gimbal, capture_image, return_home, land, get_telemetry, reset). It can be implemented against SITL when (a) the reward shape is frozen, (b) hidden-state invariants are exercised, and (c) at least one trained policy beats the scripted baseline on the geometry sim.
- Hidden defects, weather, occlusion, and quality scoring live in `simulation/` and are simulator-agnostic. SITL only needs to provide telemetry and motion; the camera quality model can stay shared.

## Minimum viable SITL feature set

When SITL is greenlit, the implementation must cover:

1. **Telemetry parity** — `get_telemetry` must populate every field the agent already sees (autopilot mode, GPS, EKF, battery, gimbal, camera, wind band, elapsed time, distance flown). Anything missing breaks `inspection_affordances`.
2. **Action parity** — every `DroneController` method must accept the same args and produce a result dict shaped like `GeometryController`'s. Mismatches will silently corrupt reward shaping.
3. **Determinism hook** — `reset(world)` must reseed the simulator from `world.scenario_seed`. Reproducibility is load-bearing for tests and benchmarks.
4. **Hidden state isolation** — SITL must not read or write the verifier-only fields on `EpisodeWorld` (`hidden_defects`, `true_asset_state`, `hidden_weather_details`).
5. **Optional test marker** — SITL tests must register under a `pytest.mark.sitl` marker so they're skipped by default. The geometry path must remain the default `pytest` run.

## What `DroneKitSITLController` does today

Every method calls `self._not_ready()` which raises a `NotImplementedError` with a pointer back to `GeometryController`. This is intentional: it prevents the placeholder from silently masquerading as a working backend. Do not mock it out for tests; route tests through `GeometryController` instead.

## Adding a third backend

Subclass `DroneController` and pass an instance into `DroneCaptureOpsEnvironment(controller=...)`. There is no backend registry — the env constructor is the single injection point. If you need a config-driven swap, add it in `server/app.py` rather than the env, since it's a deployment concern, not a domain concern.
