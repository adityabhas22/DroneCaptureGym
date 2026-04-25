# Solar Reward Components: Attack Surface and Verifier Guards

This document lists every reward component used by SolarInspect mission scoring and explains:

1. How an agent might try to game it.
2. Which verifier or gate should block that behavior.
3. What residual risk remains.

## 1) `required_coverage`

What it rewards:

- Required thermal row coverage with valid quality.

How it can be hacked:

- Submit only RGB images and claim thermal coverage.
- Cite thermal images that do not actually show required rows.
- Cite low-quality thermal captures that are too weak to trust.

What blocks it:

- `compute_required_coverage(..., cited_only=True)` checks row visibility in cited thermal captures.
- `compute_integrity_gate` warns and caps for:
  - wrong sensor for thermal requirement,
  - missing cited row coverage,
  - low-quality cited evidence.

Residual risk:

- If geometric visibility is imperfect in simulation, edge captures may occasionally pass near threshold. Threshold tuning and stricter geometric checks can reduce this.

## 2) `issue_capture`

What it rewards:

- Capturing hidden defects with required modality evidence.

How it can be hacked:

- Claim issue completion from thermal-only evidence when RGB context is required.
- Report issue IDs not present in scenario truth.
- Use unrelated row captures as fake context.

What blocks it:

- `defect_captured` enforces per-defect checks:
  - target match,
  - required sensor,
  - quality/resolution/occlusion/view-angle constraints,
  - RGB context when required.
- `compute_issue_capture(..., cited_only=True)` for terminal scoring.
- `compute_integrity_gate` caps on unsupported issue claims and incomplete cited evidence.

Residual risk:

- Partial credit (0.6) is intentional for thermal-only evidence; curriculum may need to ensure policy still prefers full completion.

## 3) `evidence_success`

What it rewards:

- Mission-level completion signal combining coverage, issue capture, and terminal safety conditions.

How it can be hacked:

- Farm coverage and issue progress but avoid true mission closure.
- Submit a report without meeting return-home or battery closure requirements.

What blocks it:

- `compute_evidence_success` includes `terminal_safety` term.
- Terminal scoring uses cited coverage/issue values after submission.
- `safety_gate` can cap outcomes if closure conditions are unsafe.

Residual risk:

- A policy may optimize for moderate evidence success without maximizing efficiency; this is expected and handled by additional terms.

## 4) `operational_efficiency`

What it rewards:

- Completing missions without excess distance, photos, time, or battery use.

How it can be hacked:

- Rush low-quality completion and exploit easy scenario geometry.
- Overcapture then rely on strong outcome terms to mask inefficiency.

What blocks it:

- `compute_operational_efficiency` uses completion gating tied to evidence success.
- Efficiency penalties scale with over-budget distance/photo/time/battery usage.

Residual risk:

- Reference budgets are heuristic. Domain expansions may require recalibration.

## 5) `grounded_report`

What it rewards:

- Evidence pack quality and traceability.

How it can be hacked:

- Fabricate photo IDs.
- Write convincing prose without evidence linkage.
- Omit open items when coverage is incomplete.
- Claim safety outcomes that telemetry does not support.

What blocks it:

- `validate_evidence_report` checks:
  - cited IDs exist,
  - requirement and issue linkage,
  - anomaly mention coverage,
  - open-item accuracy,
  - safety note consistency.
- `compute_integrity_gate` imposes hard caps for fabricated or contradictory report claims.

Residual risk:

- Free-text summaries can still be vague while formally valid. More structured fields reduce this risk.

## 6) `process_reward`

What it rewards:

- Useful intermediate behaviors (new coverage, meaningful RGB context, quality recovery, return-home transition, inspection discipline).

How it can be hacked:

- Spam low-impact tool calls to accumulate reward.
- Repeat captures hoping to farm tiny increments.

What blocks it:

- `_process_reward` only gives bonuses on concrete state improvements.
- Total process reward is capped at 0.10.
- Repeated harmless calls do not increase it.

Residual risk:

- Minor action-sequence gaming remains possible but bounded by cap and penalties.

## 7) `penalties`

What it does:

- Subtracts reward for bad behavior and low-value repetition.

How it can be hacked:

- Attempt near-duplicate captures that technically differ just enough to avoid duplicate detection.
- Exploit gaps between useful recapture and redundant recapture definitions.

What blocks it:

- `_penalties` adds costs for:
  - invalid format/action counts,
  - low-value redundant captures (`compute_photo_value < 0.05` with same sensor/targets),
  - failed return-home at mission end,
  - empty evidence submission.

Residual risk:

- Duplicate definition is intentionally conservative to avoid punishing useful recaptures.

## 8) `safety_gate`

What it does:

- Hard-caps terminal reward by severity of safety failures.

How it can be hacked:

- Try to earn high mission outcomes while accepting unsafe shortcuts.
- Intentionally end mission before return-home while retaining evidence score.

What blocks it:

- `compute_safety_gate` caps based on violations and ending state:
  - collision, no-fly, privacy, unsafe altitude, battery exhaustion,
  - timeout away from home,
  - must-return-home not satisfied,
  - repeated invalid actions.

Residual risk:

- New violation categories must be added explicitly; otherwise they cannot cap score.

## 9) `integrity_gate`

What it does:

- Hard-caps terminal reward by evidence integrity and report truthfulness.

How it can be hacked:

- Hallucinate IDs, overstate requirement completion, or claim unsupported issues.
- Submit safety notes that misrepresent telemetry state.

What blocks it:

- `compute_integrity_gate` enforces caps for:
  - fake photo IDs,
  - no-capture report submissions,
  - satisfied requirements lacking real evidence,
  - wrong-sensor citations,
  - low-quality citations,
  - unsupported issue claims,
  - incomplete cited issue evidence,
  - telemetry-contradicting safety notes.

Residual risk:

- Integrity checks rely on scenario truth and parser assumptions. Any new report schema fields must be integrated into citation extraction.

## 10) `value_per_photo` (diagnostic)

What it measures:

- Average marginal utility of captures (new coverage, new issues, RGB context, quality improvement, report usefulness).

How it can be hacked:

- Keep capture set small and cite almost every image to inflate perceived usefulness.

What blocks it:

- This metric is not part of terminal formula and cannot directly inflate `total`.
- Redundancy penalty and efficiency components counter low-value overcapture behavior.

Residual risk:

- As a diagnostic metric, it can be misleading if read alone.

## 11) Legacy Mirror Fields

Fields:

- `target_coverage` (mirror of `required_coverage`)
- `defect_visibility` (mirror of `issue_capture`)
- `route_efficiency` (mirror of `operational_efficiency`)
- `report_grounding` (mirror of `grounded_report`)

How they can be hacked:

- Same as their source components.

What blocks them:

- Same verifier paths as the source components.

Residual risk:

- None beyond source component risk; these are aliases.

## 12) High-Level Guarantee

A mission cannot get a high terminal score unless all three are true:

1. It actually captures required evidence.
2. It cites that evidence correctly in the report.
3. It remains within safety and integrity constraints.

This is the intended anti-gaming contract for SolarInspect reward learning.
