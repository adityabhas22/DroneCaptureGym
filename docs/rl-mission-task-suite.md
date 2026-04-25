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

## Implemented Task Benchmark

The task catalog is intentionally kept to 30 mechanically distinct missions. A
task should stay in the benchmark only when it changes the measurable simulator
state, verifier expectations, reward tradeoff, or optimal policy.

Current task groups:

- Core loop: `basic_thermal_survey`, `anomaly_confirmation`, `multi_anomaly_triage`, `no_anomaly_clearance`
- Resource control: `low_battery_inspection`, `capture_efficiency_discipline`, `thermal_only_anomaly_skip_rgb`, `single_row_reinspection`, `required_rows_subset_priority`
- Capture quality and camera use: `inspect_recapture_quality_loop`, `zoom_required_long_standoff`, `edge_row_quality_bar`, `diode_fault_needs_close_thermal`
- Safety and routing: `obstacle_detour_inspection`, `compound_safety_corridor`, `multi_anomaly_routing_under_obstacle`, `permanent_occlusion_coverage`, `substation_adjacency_caution`
- Privacy and false positives: `privacy_zone_capture`, `soft_privacy_capture_positioning`, `true_false_anomaly_discrimination`, `no_defect_with_glare_artifact`, `partial_blocked_anomaly_honest_report`
- Domain-specific anomaly physics: `pid_multi_row_pattern`, `bird_soiling_explanation`, `vegetation_edge_encroachment`
- Reward/report strategy: `honest_partial_report_open_items`, `strict_severity_weighted_triage`, `prioritized_triage_under_constraint`, `audit_grade_strict_grounding`

Removed low-value reskins are not part of the benchmark: `bad_weather_recapture`,
`safety_constrained_route`, `sparse_evidence_trap`,
`closeup_resolution_challenge`, `edge_row_focus`, `return_home_compliance`,
`limited_steps_rapid_survey`, `report_grounding_audit`,
`string_outage_survey`, `cracked_glass_closeup`, `low_contrast_recapture`,
`boundary_aware_closeup`, and `adaptive_battery_reserve`.

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

Run the deterministic task benchmark:

```bash
python -m training.run_suite --suite solar_tasks --policy scripted
```

Run the deterministic scripted helper over the task catalog:

```bash
python examples/run_task_suite.py
```

Run the original scripted baseline:

```bash
python examples/run_scripted_agent.py
```

## Files Added Or Updated

Key files:

- `dronecaptureops/tasks/solar_tasks.py`
- `dronecaptureops/generation/suites.py`
- `dronecaptureops/domains/solar.py`
- `tests/test_solar_tasks.py`
- `tests/test_solar_task_benchmark.py`

## Completion Status

The corrected implementation is complete end to end:

- 30 task IDs are available.
- All tasks reset deterministically.
- All tasks preserve OpenEnv action compatibility.
- All tasks have visible mission instructions and success criteria.
- Hidden defects remain internal until valid sensing.
- The `solar_tasks` suite runs through the harness with random and scripted policies.
- Tests cover task generation, hidden-state protection, exact anomaly/RGB pairing, partial-report rejection, privacy capture safety, task suite execution, and mechanical assertions for the corrected task catalog.
