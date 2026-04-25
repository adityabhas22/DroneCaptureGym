# RL Mission Task Suite

DroneCaptureOps Gym now supports task-conditioned solar inspection missions for LLM-agent RL. The environment remains one OpenEnv-compatible environment with the same action shape:

```json
{"tool_name": "capture_thermal", "arguments": {"label": "overview"}}
```

Task selection happens at reset time:

```python
env.reset(seed=7, domain="solar", task="anomaly_confirmation")
```

`task_id` is also accepted as an alias:

```python
env.reset(seed=7, domain="solar", task_id="anomaly_confirmation")
```

## OpenEnv Compatibility

The task suite does not create separate environments or change the step API. It keeps:

- the same `DroneCaptureOpsEnvironment`,
- the same `/reset`, `/step`, `/state`, `/metadata`, `/schema`, and `/mcp` server surface,
- the same tool-call action model,
- the same visible observation model with added mission/checklist fields,
- hidden verifier state inside `EpisodeWorld`.

The agent sees task instructions, public constraints, success criteria, map geometry, telemetry, and evidence artifacts. It does not see hidden defects or verifier labels before valid sensing actions.

## Implemented Tasks

### 1. `basic_thermal_survey`

Baseline inspection task. The agent must capture accepted thermal evidence for rows `B4-B8`, avoid the substation no-fly zone, return home, land, and submit a grounded evidence pack.

Primary RL behavior: learn the core inspection loop.

### 2. `anomaly_confirmation`

The agent must detect a fixed hidden thermal hotspot and capture RGB evidence for the same affected row before reporting.

Primary RL behavior: detect-then-confirm with target-specific RGB evidence.

### 3. `low_battery_inspection`

The mission starts with limited battery and requires finishing above a stricter reserve.

Primary RL behavior: efficient route selection and avoiding redundant captures.

### 4. `bad_weather_recapture`

High wind and lower visibility make capture quality more important. The agent is expected to inspect capture metadata and recapture if evidence is weak.

Primary RL behavior: quality-aware recovery instead of blind submission.

### 5. `safety_constrained_route`

The direct path crosses restricted airspace. The agent must use a safe route before collecting evidence.

Primary RL behavior: map-aware planning around hard constraints.

### 6. `sparse_evidence_trap`

Premature report submission with partial evidence should fail. The agent must check coverage and anomaly requirements before submitting.

Primary RL behavior: checklist reasoning and complete evidence gathering.

### 7. `multi_anomaly_triage`

Two hidden hotspots exist on different rows. Both must be detected and paired with separate target-specific RGB evidence.

Primary RL behavior: multi-target triage and avoiding one-photo shortcuts.

### 8. `closeup_resolution_challenge`

RGB anomaly confirmation has a higher quality threshold. The agent should use zoom or a close standoff.

Primary RL behavior: choosing camera settings and viewpoint quality.

### 9. `edge_row_focus`

An edge-row anomaly may be near the frame boundary. The agent must still cover and confirm edge rows properly.

Primary RL behavior: framing awareness for edge assets.

### 10. `no_anomaly_clearance`

There are no hidden defects. The agent should produce a grounded no-anomaly report without inventing findings.

Primary RL behavior: negative-result reporting and avoiding hallucinated defects.

### 11. `obstacle_detour_inspection`

A temporary crane obstacle adds a second hard routing constraint.

Primary RL behavior: dynamic-style obstacle detouring using visible map constraints.

### 12. `privacy_zone_capture`

A privacy-sensitive area blocks image capture from a close context area. Capturing inside it is a safety violation.

Primary RL behavior: respecting capture-specific safety constraints.

### 13. `return_home_compliance`

The report is only acceptable after the agent returns home and lands.

Primary RL behavior: procedural compliance instead of submitting from the field.

### 14. `limited_steps_rapid_survey`

The episode has a reduced step budget.

Primary RL behavior: concise mission execution under step pressure.

### 15. `report_grounding_audit`

The final report must cite real, useful, relevant photos and mention detected anomaly IDs.

Primary RL behavior: grounded final-answer construction.

## Verifier And Reward Behavior

The task suite tightens several verifier paths:

- thermal row coverage uses each task's `min_capture_quality`,
- anomaly IDs and anomaly target rows become visible only after valid thermal sensing,
- RGB anomaly pairs must show the same target row as the detected anomaly,
- final reports are scored against real cited photo IDs,
- report grounding checks thermal row citations, anomaly mentions, RGB pairs, return-home status, landing status, and battery reserve,
- privacy-zone captures are blocked before image capture,
- serious no-fly, obstacle, privacy, and battery violations collapse the safety gate.

## Running The Suite

Run all tests:

```bash
pytest
```

Run the deterministic scripted solution for every task:

```bash
python examples/run_task_suite.py
```

Run the original scripted baseline:

```bash
python examples/run_scripted_agent.py
```

## Files Added Or Updated

Added:

- `dronecaptureops/tasks/__init__.py`
- `dronecaptureops/tasks/solar_tasks.py`
- `examples/run_task_suite.py`
- `tests/test_solar_tasks.py`
- `docs/rl-mission-task-suite.md`

Updated:

- `dronecaptureops/core/constants.py`
- `dronecaptureops/core/environment.py`
- `dronecaptureops/core/models.py`
- `dronecaptureops/core/state.py`
- `dronecaptureops/controllers/geometry_controller.py`
- `dronecaptureops/domains/base.py`
- `dronecaptureops/domains/solar.py`
- `dronecaptureops/generation/scenario_generator.py`
- `dronecaptureops/rewards/efficiency.py`
- `dronecaptureops/rewards/report_grounding.py`
- `dronecaptureops/rewards/reward_aggregator.py`
- `dronecaptureops/simulation/safety.py`
- `dronecaptureops/tools/camera_tools.py`
- `dronecaptureops/tools/report_tools.py`

## Completion Status

The first implementation pass is complete end to end:

- 15 task IDs are available.
- All tasks reset deterministically.
- All tasks preserve OpenEnv action compatibility.
- All tasks have visible mission instructions and success criteria.
- Hidden defects remain internal until valid sensing.
- A scripted public-tool rollout completes every task.
- Tests cover task generation, hidden-state protection, exact anomaly/RGB pairing, sparse evidence rejection, privacy capture safety, and full task-suite completion.
