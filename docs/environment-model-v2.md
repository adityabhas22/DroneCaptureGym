# Environment Model V2

DroneCaptureOps Gym models mission-level drone inspection, not motor control. The LLM sees high-level tools and structured observations; the simulator owns autopilot state, safety checks, camera/gimbal behavior, battery use, and hidden verifier truth.

## Visible State

Observations expose:

- mission brief and public checklist progress,
- DroneKit-style telemetry: autopilot mode, armed state, EKF/GPS readiness, heartbeat age, pose, velocity, attitude, battery, command link, rangefinder, gimbal, camera, wind band, elapsed mission time, and distance flown,
- generic site map with inspectable assets, airspace zones, and candidate viewpoints,
- evidence artifacts created by real capture tools,
- recent warnings and a compact `state_summary` for LLM-friendly inspection.

The OpenEnv action shape remains:

```json
{"tool_name": "capture_thermal", "arguments": {"label": "overview"}}
```

## Internal State

`EpisodeWorld` keeps verifier-only fields out of observations:

- true defect map and hidden asset condition,
- hidden weather details and future obstacle schedules,
- verifier evidence requirements,
- action and observation logs.

Only valid sensing actions can create evidence artifacts. Final reports must cite real artifact IDs; hidden defects are never inserted into the observation or visible state.

## Solar V2 Domain

Solar rows are represented as generic `InspectableAsset` objects with:

- asset geometry and viewing normal,
- required modalities,
- safe standoff bands,
- visibility tags and public notes.

The map also includes `AirspaceZone` constraints and named `Viewpoint` candidates. This keeps the solar domain concrete while preserving the same interface for industrial, construction, bridge, and other future inspection sites.

## DroneKit Mapping

The geometry backend mirrors the fields a DroneKit/ArduPilot adapter would use later:

- `AutopilotState` maps to `Vehicle.mode`, `Vehicle.armed`, `Vehicle.is_armable`, `Vehicle.system_status`, `Vehicle.ekf_ok`, heartbeat, and command status.
- `GpsState`, `BatteryState`, `RangefinderState`, `Velocity3D`, and `Attitude` map to DroneKit vehicle attributes populated from MAVLink streams.
- `GimbalState` and `CameraState` map to MAVLink gimbal/camera commands such as pitch/yaw control, ROI pointing, camera source, zoom, focus, and image capture.

The current backend is still deterministic and CPU-light. It does not launch SITL, render images, or run learned perception.
