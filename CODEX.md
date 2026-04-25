# DroneCaptureOps Gym Engineering Guide

## Project Mission

DroneCaptureOps Gym is an OpenEnv-compatible RL environment for training LLM agents to conduct aerial inspection missions through high-level drone and camera tools. The agent plans viewpoints, moves to safe waypoints, adjusts camera and gimbal settings, captures RGB and thermal evidence, reviews capture quality, revisits missed targets, returns home safely, and submits a grounded evidence pack.

The core benchmark is active visual inspection, not raw drone navigation. Movement exists to collect complete, high-quality evidence.

## Non-Goals

- Do not implement raw motor control.
- Do not expose hidden verifier state, hidden defects, or expected answers to the agent.
- Do not optimize only for shortest path.
- Do not create reward functions that are easy to game.
- Do not couple the environment to a single simulator backend.
- Do not hard-code domain-specific logic into generic environment classes.

## Engineering Principles

### SOLID

- Single Responsibility: each module and class should have one clear reason to change.
- Open/Closed: add new domains, controllers, tools, and reward components without editing core orchestration logic.
- Liskov Substitution: every controller backend must satisfy the same `DroneController` interface.
- Interface Segregation: avoid forcing controllers, domains, or reward classes to implement unused methods.
- Dependency Inversion: core environment logic depends on abstractions, not concrete backends.

### DRY

- Avoid duplicate reward logic.
- Avoid duplicate geometry or camera calculations.
- Centralize action schema definitions.
- Centralize constants and thresholds.
- Prefer reusable helpers over copy-pasted calculations.

### Extensibility

The codebase should make it straightforward to add domains such as solar, construction, bridge, and industrial inspections; controller backends; camera and sensor models; reward components; scenario generators; OpenEnv tools; training scripts; and demos.

### Testability

Important logic must be testable without launching a heavy simulator. Tests should cover reset determinism, action validation, safety checks, reward components, hidden-state protection, final evidence-pack verification, and domain scenario generation.

### Separation of Concerns

Keep separate the OpenEnv interface, drone controller, world simulation, camera simulation, reward computation, domain scenario logic, training scripts, and demo scripts.

### Safety

The environment must enforce safety constraints before executing actions. Safety checks include no-fly zone violations, collision or obstacle violations, battery exhaustion, unsafe altitude, invalid gimbal angle, invalid waypoint, privacy-zone capture, repeated invalid actions, and failure to return home when required.

### Reproducibility

Use deterministic seeds. Every episode should be reproducible from domain, scenario seed, environment config, and agent action log.

### Observability

Every episode should log the action sequence, observations, reward breakdown, safety violations, captures taken, checklist status, and final evidence pack.

## Current Structure

```text
dronecaptureops/
  core/          OpenEnv environment, typed models, action validation
  controllers/   backend-neutral DroneController plus geometry and SITL adapters
  simulation/    geometry, camera, battery, safety, and world state
  domains/       domain scenario builders, starting with solar
  rewards/       composable reward components and aggregation
  tools/         high-level action registry and tool handlers
  generation/    seeded scenario generation entrypoints
  server/        package-local server helpers
  utils/         shared math and serialization helpers
server/          OpenEnv HTTP app entrypoint
training/        RL and evaluation scaffolds
examples/        runnable random/scripted episodes
tests/           fast deterministic pytest coverage
```

## Implementation Guidance

Keep the MVP geometry-first. Do not add photorealistic rendering or full DroneKit integration until the fast simulator, reward shape, and evidence verification are stable. `DroneKitSITLController` should remain an adapter behind the same controller interface.
