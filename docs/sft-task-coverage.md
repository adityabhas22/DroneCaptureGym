# SFT task coverage — what we train on, what we hold out, why

## TL;DR

- **Train on 38 of 45 tasks**. Held-out 7 cover one mechanic family each (cross-task generalisation eval).
- **All 45 task IDs are now reference-solvable** by the spec-aware scripted solver in `examples/run_task_suite.py`. SFT data covers the full mechanic distribution.
- **Anti-overfit rules unchanged**: LoRA r=16, 3 epochs max, train/val split by seed inside each train task, early stopping on val_loss.

This replaces the old "20 oracle-solvable + 25 RL-only" split. With the spec-aware solver, every task can produce a clean reference trajectory, so RL only has to *refine* — not *discover* — the mechanic vocabulary.

## Why this matters for SFT → PPO

A tool used by an unseen-during-SFT mechanic has near-zero prior under the SFT-warm-started policy. Random exploration in an 18-tool action space rarely surfaces the right tool for the right context, so PPO essentially never recovers tools the model never saw. Specifically: `request_route_replan`, `mark_target_inspected`, severity-weighted submission patterns, and partial-report `open_items` are all behaviours the previous oracle never demonstrated.

By giving SFT exposure to all 7 mechanic families, PPO inherits a policy with non-zero probability mass on the right tools at the right moments — its job becomes "make the existing strategy more reward-efficient" rather than "discover new strategies from scratch."

## The 38 train tasks, by mechanic axis

| Axis | Tasks |
|---|---|
| Baseline coverage | `basic_thermal_survey`, `commissioning_acceptance_survey` |
| Battery / efficiency | `low_battery_inspection`, `capture_efficiency_discipline`, `quality_vs_efficiency_tradeoff` |
| Anomaly confirmation | `anomaly_confirmation`, `multi_anomaly_triage`, `multi_issue_one_rgb_context`, `thermal_only_anomaly_skip_rgb`, `thermal_only_fast_clearance` |
| Defect taxonomy | `diode_fault_needs_close_thermal`, `bird_soiling_explanation`, `vegetation_edge_encroachment`, `pid_multi_row_pattern` |
| False-positive discrimination | `true_false_anomaly_discrimination`, `no_defect_with_glare_artifact`, `glare_angle_experiment` |
| Negative report | `no_anomaly_clearance` |
| Airspace / safety | `privacy_zone_capture`, `substation_adjacency_caution`, `soft_privacy_capture_positioning`, `compound_safety_corridor`, `obstacle_detour_inspection`, `permanent_occlusion_coverage`, `multi_anomaly_routing_under_obstacle` |
| Strict grounding | `audit_grade_strict_grounding`, `warranty_claim_evidence_pack` |
| Required-rows scoping | `single_row_reinspection`, `required_rows_subset_priority`, `post_repair_verification`, `minimum_evidence_for_dispatch` |
| Severity-weighted triage | `prioritized_triage_under_constraint`, `low_severity_ignore_under_budget` |
| Honest partial reports | `partial_blocked_anomaly_honest_report`, `operator_abort_under_safety_pressure` |
| Quality-loop / recapture | `inspect_recapture_quality_loop` |
| Long-standoff zoom | `zoom_required_long_standoff` |
| Safe-dogleg return | `blocked_return_path_requires_safe_dogleg` |

Total: 38 distinct task IDs, 13 distinct mechanic axes.

## The 7 held-out tasks (eval only — never in training set)

One per mechanic family the v2 suite introduced. Each held-out task pairs with a related-but-distinct train task so the model has seen the broader axis without same-task data leakage.

| Held-out task | Mechanic family | Closest train pair |
|---|---|---|
| `scheduled_crane_window_wait_or_detour` | dynamic-obstacle scheduling | `obstacle_detour_inspection` |
| `honest_partial_report_open_items` | honest partial reporting | `partial_blocked_anomaly_honest_report` |
| `strict_severity_weighted_triage` | severity-weighted triage | `prioritized_triage_under_constraint` |
| `edge_row_quality_bar` | tight-quality framing | `warranty_claim_evidence_pack` |
| `privacy_safe_alternate_evidence` | privacy-safe alternates | `soft_privacy_capture_positioning` |
| `return_margin_decision_point` | return-margin decision | `low_battery_inspection` |
| `route_replan_when_primary_viewpoint_blocked` | route-replan with extra VPs | `permanent_occlusion_coverage` |

## Generation policy — four strategy variants

The spec-aware solver runs in four strategy modes; all are verifier-passing (180/180 task×strategy combinations) and produce genuinely different action sequences. Listing all four in the data-gen rotation gives the model multiple valid solutions per task plus explicit failure-recovery demonstrations.

| Strategy | Behaviour |
|---|---|
| `s0` (careful) | Explicit `set_camera_source`; severity-weighted RGB; `mark_target_inspected` after RGB; `inspect_capture` after each capture; baseline summary |
| `s1` (streamlined) | Skip warm-up calls; row-position RGB ordering; compact ops-log summary; `hover` wait-strategy on scheduled-obstacle tasks |
| `s2` (diagnostic) | `get_site_map` + `get_telemetry` + `list_assets` + `get_mission_checklist`; `estimate_return_margin`; proactive `request_route_replan` when blocked; narrative summary |
| `s3` (recovery) | Like `s0` PLUS injects ONE `set_gimbal(pitch=-120)` invalid-argument call before a thermal capture, env returns `invalid_gimbal_pitch:-120.0`, drone state unchanged, next action retries with a valid pitch. Trains the model to read `obs.error` and emit a corrective tool call. Skipped on tight-budget tasks. |

All four are listed in `training/configs/sft_default.yaml` under `policies:` so each (task, seed) produces up to four trajectories with different tool subsets, ordering, summary phrasings, and (for s3) error-recovery cycles.

### What the recovery pattern looks like in trajectories

Sample assistant/user message pair from a strategy-3 trajectory:

```
... assistant: set_gimbal(pitch_deg=-120.0, yaw_deg=0)
... user: # step 6 | phase: survey | steps_remaining: 34 | last_error: invalid_gimbal_pitch:-120.0
... assistant: set_gimbal(pitch_deg=-56, yaw_deg=0)   ← model corrects
... user: # step 7 | phase: survey | steps_remaining: 33
```

The model trains on this pattern: "when the previous action's user-response shows `last_error:`, the next action should be a corrective retry with valid arguments." Without strategy 3, the model never sees an env error in its training data and would have no prior on how to recover.

## Anti-overfit rationale

A common failure mode for SFT → PPO: the SFT corpus is so dense and so close to the reward-maximising trajectory that the policy distribution collapses onto the solver's exact action sequences. PPO then has near-zero entropy to work with and can't explore.

We mitigate three ways:

1. **LoRA r=16 + 3-epoch cap** (`training/configs/sft_train_default.yaml`) — limits memorisation surface to ~1% of model parameters.
2. **Hold out 7 tasks entirely** — forces the model to generalise across mechanic axes, not just within-seed.
3. **Reference solver, not oracle** — the spec-aware solver makes deliberate, sometimes-suboptimal choices (e.g., transit via far-east when north blocked) rather than perfect ones. RL has room to find better paths.

## Dataset shape (verified)

With `seeds_per_task: 12`, four policy strategies (`s0`, `s1`, `s2`, `s3`), `require_success: true`, dedup on:

| Metric | Value |
|---|---|
| Tasks in train set | 39 |
| Tasks held out | 6 |
| Raw trajectories generated | 2160 (45 tasks × 12 seeds × 4 strategies) |
| Successful | 2160 (100%) |
| Kept after dedup | 1578 |
| Train JSONL records | 1368 |
| Held-out JSONL records | 210 |
| **Records with explicit env-error recovery (s3)** | **258** (~19% of train) |
| Unique tool-name sequences | **120** |
| Unique full-message sequences | **144** |
| Mean reward (all strategies) | 0.998–0.999 |

### Per-strategy summary

| Strategy | Records | Mean reward | What it teaches |
|---|---|---|---|
| s0 careful | 366 | 0.999 | Explicit per-capture setup, target-inspection signal |
| s1 streamlined | 366 | 0.999 | Efficient ordering, hover wait-strategy |
| s2 diagnostic | 378 | 0.999 | Info-gathering tools, proactive replan |
| s3 recovery | 258 | 0.998 | Reading `obs.error`, retrying with valid arguments |

## Tool coverage (19 of 22 env tools demonstrated)

| Used | Count | Used | Count |
|---|---|---|---|
| `fly_to_viewpoint` | 5,110 | `mark_target_inspected` | 544 |
| `set_gimbal` | 3,200 | `get_mission_checklist` | 504 |
| `capture_thermal` | 2,060 | `list_assets` | 258 |
| `capture_rgb` | 1,470 | `get_site_map` | 258 |
| `inspect_capture` | 1,520 | `get_telemetry` | 258 |
| `takeoff` | 1,110 | `estimate_return_margin` | 258 |
| `return_home` | 1,110 | `set_zoom` | 210 |
| `land` | 1,110 | `hover` | 80 |
| `submit_evidence_pack` | 1,110 | `request_route_replan` | 30 |
| `set_camera_source` | 730 |  |  |

Each strategy demonstrates a distinct tool subset:
- `s0` adds explicit `set_camera_source` + `mark_target_inspected`
- `s1` adds `hover` (wait-strategy on scheduled obstacles)
- `s2` adds `get_site_map` + `get_telemetry` + `list_assets` + `estimate_return_margin` + active `request_route_replan`

**Three tools remain unused**: `estimate_view`, `move_to_asset`, `point_camera_at`. All three are functional alternates to tools already heavily demonstrated (`set_gimbal` covers aiming; `fly_to_viewpoint` covers `move_to_asset`; explicit quality checks replace `estimate_view`). Coverage of the *necessary* tool surface is complete; the missing tools are stylistic alternatives PPO can refine into if useful.

### Diversity vs the prior single-strategy corpus

| Diversity metric | Single strategy | Three strategies | Improvement |
|---|---|---|---|
| Records | 356 | 1080 | 3.0× |
| Unique tool-name sequences | 38 (1/task) | 92 | 2.4× |
| Unique full-message sequences | 38 | 114 | 3.0× |
| Unique submit summaries | ~38 | 98 | 2.6× |
| Per-task tool-shape variants | 1 | 1–3 | up to 3× |

For a handful of tight-budget tasks (`single_row_reinspection`, `minimum_evidence_for_dispatch`, etc.), the tool-name sequence is identical across strategies because the budget can't fit alternative tool calls. But the *full-content* sequences still differ via summary phrasing (3 distinct summaries per tight task) — and that's what the model trains on, not just tool names.

### Per-strategy mean reward (sanity check on validity)

| Strategy | Records | Mean reward |
|---|---|---|
| s0 careful | 356 | 0.998 |
| s1 streamlined | 356 | 0.999 |
| s2 diagnostic | 368 | 0.999 |

All three strategies cleanly pass the integrity gate. The slight per-strategy variation (0.998 vs 0.999) reflects edge cases in the careful strategy where the longer trajectory exhausts a tight step budget; not a quality issue.
