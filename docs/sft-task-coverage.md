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

## Generation policy

`spec_aware_scripted` (the solver wrapped as a Policy in `dronecaptureops/agent/spec_aware_policy.py`) is the only policy in the SFT mix. With 45 distinct mechanic types × 12 seeds, the diversity is coming from *task variety* rather than *strategy variety on the same task*. Adding a second policy that produces a different strategy for the same task would dilute, not enrich, the corpus. RL is the right tool to discover alternative strategies.

## Anti-overfit rationale

A common failure mode for SFT → PPO: the SFT corpus is so dense and so close to the reward-maximising trajectory that the policy distribution collapses onto the solver's exact action sequences. PPO then has near-zero entropy to work with and can't explore.

We mitigate three ways:

1. **LoRA r=16 + 3-epoch cap** (`training/configs/sft_train_default.yaml`) — limits memorisation surface to ~1% of model parameters.
2. **Hold out 7 tasks entirely** — forces the model to generalise across mechanic axes, not just within-seed.
3. **Reference solver, not oracle** — the spec-aware solver makes deliberate, sometimes-suboptimal choices (e.g., transit via far-east when north blocked) rather than perfect ones. RL has room to find better paths.

## Dataset shape (verified)

Run `python -m training.generate_sft_data` for actuals. With `seeds_per_task: 12`, single `spec_aware_scripted` policy, `require_success: true`, dedup on:

| Metric | Value |
|---|---|
| Tasks in train set | 38 |
| Tasks held out | 7 |
| Raw trajectories generated | 540 (45 tasks × 12 seeds) |
| Successful (`require_success`) | 540 (100%) |
| Kept after dedup | 426 |
| Train JSONL records | 356 |
| Held-out JSONL records | 70 |
| After train/val split (val_seed_fraction=0.20) | 283 train / 73 val |
| All trajectories successful | 426/426 (mean reward 1.000) |

Per-task coverage in train ranges from 2 (when solver is deterministic enough that all 12 seeds produce identical traces — bird_soiling, diode_fault, recapture-loop) to 10 (most tasks). Mean ~9.4 per task.

The lower per-task counts on a handful of tasks reflect that the solver is procedurally identical across seeds when defect placement happens to be invariant. This is a feature, not a bug — duplicate traces would be removed by dedup anyway, and the SFT pipeline is robust to imbalanced per-task counts (each task contributes its mechanic signal once; PPO refines).
