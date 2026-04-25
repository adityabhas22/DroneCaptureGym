# SFT task coverage — what we train on, what we hold out, why

## TL;DR

- **Train on 16 of 45 tasks** (the oracle-solvable subset minus 4 held-out).
- **Hold out 4** for cross-task generalisation eval.
- **25 tasks are RL-only** — the legacy oracle can't solve them. Those mechanics are reached by PPO exploration, not by imitation.

This is deliberate. The 16 SFT tasks cover the major mechanic axes — battery, multi-anomaly, false-positive, sensor choice, no-defect, airspace, audit grounding — so the model lands at SFT end with the *format* and *broad strategy* nailed, with room to discover the harder mechanics under PPO.

## The 16 train tasks, by mechanic axis

| Axis | Task | Mechanic it teaches |
|---|---|---|
| Baseline | `basic_thermal_survey` | Standard 5-row thermal sweep + return-home |
| Battery / efficiency | `low_battery_inspection` | Operate inside 45% init / 35% reserve constraint |
| Anomaly confirmation | `anomaly_confirmation` | Pair thermal detection with same-row RGB |
| Anomaly confirmation | `multi_issue_one_rgb_context` | One RGB photo can support two adjacent anomalies |
| Anomaly confirmation | `thermal_only_anomaly_skip_rgb` | Skip RGB when defect type doesn't require it |
| Defect taxonomy | `diode_fault_needs_close_thermal` | Bypass-diode invisible from overview, requires close thermal |
| Defect taxonomy | `bird_soiling_explanation` | Soiling thermal anomaly paired with RGB showing cause |
| Defect taxonomy | `vegetation_edge_encroachment` | Edge-row shadow needs oblique gimbal + RGB |
| Defect taxonomy | `pid_multi_row_pattern` | PID degradation across B5–B7 rows |
| False-positive discrimination | `true_false_anomaly_discrimination` | Discriminate real anomaly vs glare via gimbal pitch |
| False-positive discrimination | `no_defect_with_glare_artifact` | Verify glare carefully, don't report it |
| Negative report | `no_anomaly_clearance` | Clean no-defect report without hallucinating issues |
| Negative report | `commissioning_acceptance_survey` | Strict-grounding clean block with full coverage |
| Airspace / safety | `privacy_zone_capture` | Capture from outside privacy zone (not inside) |
| Airspace / safety | `substation_adjacency_caution` | Extra safety buffer beyond hard NFZ |
| Strict grounding | `warranty_claim_evidence_pack` | 0.70 RGB + 0.90 grounding thresholds |

## The 4 held-out tasks (eval only — never in training set)

| Task | Why held out |
|---|---|
| `multi_anomaly_triage` | Cross-task generalisation: separate-RGB-per-anomaly across multiple targets |
| `audit_grade_strict_grounding` | Tests learned strict-grounding behaviour without same-task data leakage |
| `obstacle_detour_inspection` | Cross-task generalisation: safety-aware alternative routing |
| `glare_angle_experiment` | Tests learned discrimination behaviour without same-task data leakage |

These were chosen so each held-out task pairs with a *related but distinct* train task — the model has seen the broader mechanic, but not this specific instance.

## The 25 RL-only tasks (no SFT data)

The legacy `TaskOraclePolicy` solves the 20 tasks above. The remaining 25 introduce mechanics the oracle wasn't designed for: scheduled obstacles, viewpoint replanning, partial-blocked reporting, post-repair verification, return-margin decisions, single-row scope restrictions, etc.

Listing them so it's explicit they exist and are not silently ignored:

```
inspect_recapture_quality_loop, compound_safety_corridor,
honest_partial_report_open_items, zoom_required_long_standoff,
edge_row_quality_bar, soft_privacy_capture_positioning,
multi_anomaly_routing_under_obstacle, single_row_reinspection,
strict_severity_weighted_triage, permanent_occlusion_coverage,
prioritized_triage_under_constraint, capture_efficiency_discipline,
partial_blocked_anomaly_honest_report, required_rows_subset_priority,
return_margin_decision_point, route_replan_when_primary_viewpoint_blocked,
scheduled_crane_window_wait_or_detour, minimum_evidence_for_dispatch,
post_repair_verification, operator_abort_under_safety_pressure,
privacy_safe_alternate_evidence, quality_vs_efficiency_tradeoff,
thermal_only_fast_clearance, low_severity_ignore_under_budget,
blocked_return_path_requires_safe_dogleg
```

PPO is expected to discover these. The eval harness reports per-task success rates, so we'll see which of the 25 the model picks up purely from RL.

## Anti-overfit rationale

A common failure mode for SFT-then-RL: the SFT corpus is so dense and so close to the reward-maximising trajectory that the policy distribution collapses onto the oracle's exact action sequences. PPO then has near-zero entropy to work with and can't explore. We mitigate three ways:

1. **LoRA r=16 + 3-epoch cap** (in `sft_train_default.yaml`) — limits memorisation surface.
2. **Hold-out 4 tasks entirely** — forces the model to generalise, not just look-up.
3. **Don't include tasks the oracle can't solve** — better than imitating broken trajectories. RL discovers those mechanics directly.

## Dataset shape (expected)

Run `python -m training.generate_sft_data --dry-run` for actuals. With `seeds_per_task: 15`, `policies: [task_oracle, scripted]`, and `require_success: true`:

- Train tasks: 16 × 15 seeds × 2 policies = up to 480 trajectories
- After `require_success` filter: ~280–360 (scripted has lower task coverage than oracle)
- After dedup: ~250–320 unique trajectories
- After `val_seed_fraction=0.20` split: ~200 train, ~50 val
- Held-out (separate JSONL): 4 × 15 = 60 trajectories
