# Solar Rewards Implementation (How It Actually Scores)

This document explains how the current SolarInspect reward system works in code, why the design is shaped this way, and how to read reward values during training.

The core idea is:

1. Reward mission outcomes, not button pressing.
2. Use deterministic verifiers instead of text judging.
3. Allow non-terminal shaping, but only a small amount.
4. Use hard caps to block unsafe or dishonest high scores.

## 1) Mental Model

The reward system has two phases:

1. Non-terminal phase (before submit):
   The environment returns a capped shaping reward in [0.0, 0.20].
2. Terminal phase (after `submit_evidence_pack`):
   The environment computes outcome reward and then applies safety and integrity caps.

Think of non-terminal reward as progress breadcrumbs and terminal reward as the mission grade.

## 2) Terminal Reward Formula

The final mission score uses this outcome-first formula:

```text
R_total = min(
  safety_gate,
  integrity_gate,
  0.45 * evidence_success
  + 0.20 * required_coverage
  + 0.15 * issue_capture
  + 0.10 * operational_efficiency
  + 0.10 * grounded_report
  + process_reward
  - penalties
)
```

Intuition for the weights:

- `evidence_success` (0.45) is the dominant term because it summarizes whether the mission delivered the required proof.
- `required_coverage` (0.20) ensures thermal rows are truly covered.
- `issue_capture` (0.15) rewards finding and proving hidden defects.
- `operational_efficiency` (0.10) discourages wasteful missions.
- `grounded_report` (0.10) rewards report quality and evidence linkage.

If a mission is unsafe or dishonest, `safety_gate` or `integrity_gate` clamps the final score even when the weighted sum is high.

## 3) Non-Terminal Shaping

Before a final report is submitted, `total` is not the formula above. It is a capped shaping value:

```text
shaping = clamp(
  0.10 * captured_evidence_success
  + 0.04 * captured_coverage
  + 0.04 * captured_issue
  + process_reward
  - penalties,
  0.0,
  0.20
)
```

Why this exists:

- Gives gradients early in the episode.
- Prevents agents from farming high reward before proving completion.

The `debug` payload exposes this split through:

- `terminal_submitted`
- `nonterminal_cap_applied`
- `shaping_reward`
- `raw_outcome_if_submitted`

## 4) Captured Evidence vs Cited Evidence

A major design detail is the captured/cited distinction.

- Captured metrics: what the drone has actually collected so far.
- Cited metrics: what the final report explicitly cites and can defend.

Behavior:

1. During exploration, coverage and issue values come from captured evidence.
2. At terminal scoring, those same values switch to cited-only evidence.

This prevents a common failure mode where an agent collects valid evidence but submits a report that fails to reference it correctly.

## 5) Component-by-Component Intuition

### 5.1 `required_coverage`

Definition:

- Fraction of required rows that have at least one valid thermal capture (`quality >= 0.55`).

Intuition:

- One high-quality thermal can cover multiple rows.
- Wrong sensor (RGB) does not count toward thermal requirement.

### 5.2 `issue_capture`

Definition:

- Weighted hidden-defect capture score across mission defects.
- A defect can score partial credit (0.6) when thermal proof exists but required RGB context is missing.

Intuition:

- Thermal finds the abnormality.
- RGB provides contextual confirmation when the defect requires it.
- Full score means both evidence types are present when required.

### 5.3 `evidence_success`

Definition:

```text
0.55 * required_coverage + 0.35 * issue_capture + 0.10 * terminal_safety
```

where `terminal_safety` requires final-report submission plus return-home and battery constraints.

Intuition:

- This is the mission-level "did you actually complete the contract" scalar.
- It combines coverage, defect evidence, and mission closure conditions.

### 5.4 `operational_efficiency`

Definition:

- Efficiency score gated by completion and penalized by excess distance, photos, time, and battery overuse relative to scenario-scaled reference budgets.

Intuition:

- Efficient operations matter only after meaningful progress.
- Fast but incomplete missions do not earn strong efficiency reward.

### 5.5 `grounded_report`

Definition:

- Score from `validate_evidence_report`, based on:
  - requirement-photo linkage,
  - issue-photo linkage,
  - open-item correctness,
  - safety note consistency.

Intuition:

- The report is graded as an auditable artifact, not just free text.

### 5.6 `process_reward`

Definition:

- Small cumulative bonus (capped at 0.10) for useful intermediate actions such as:
  - new thermal coverage,
  - RGB issue-context improvement,
  - recapturing poor evidence with better quality,
  - return-home completion,
  - inspecting captured photos.

Intuition:

- Encourages good workflow habits.
- Too small to dominate final scoring.

### 5.7 `penalties`

Definition:

- Deductions for invalid format/actions, low-value duplicate captures, and poor mission-end behavior.

Intuition:

- Penalizes noise and gaming without punishing genuinely useful recaptures.

### 5.8 `safety_gate`

Definition:

- Cap derived from violations and unsafe endings.

Examples:

- collision can cap at 0.0,
- no-fly violation can cap at 0.10,
- timeout away from home can cap at 0.40.

Intuition:

- Safety is non-negotiable. High evidence cannot erase severe safety failures.

### 5.9 `integrity_gate`

Definition:

- Cap derived from evidence/report integrity checks.

Examples:

- fake photo IDs,
- "satisfied" requirements without valid photos,
- thermal requirements citing RGB,
- low-quality evidence cited as definitive,
- unsupported issue claims,
- safety claims contradicting telemetry.

Intuition:

- Prevents reward from being hacked through report fabrication.

## 6) Why Deterministic Verifiers

The verifier module computes all key mission checks from simulator state and capture metadata.

Benefits:

1. Reproducible and debuggable scoring.
2. Less reward drift across model versions.
3. Clear anti-cheating hooks.

## 7) Legacy Fields Still Exposed

For compatibility with existing downstream consumers:

- `target_coverage` mirrors `required_coverage`
- `defect_visibility` mirrors `issue_capture`
- `route_efficiency` mirrors `operational_efficiency`
- `report_grounding` mirrors `grounded_report`

## 8) Evidence Pack Formats

`submit_evidence_pack` accepts both:

1. Legacy payload (`summary`, `photo_ids`, `findings`)
2. Structured payload (`evidence`, `issues_found`, `open_items`, `safety_notes`)

Both paths are normalized by verifier logic before scoring.

## 9) Practical Reading Guide for Training Curves

When `total` is low, inspect these in order:

1. `safety_gate` and `integrity_gate` (caps)
2. `required_coverage` and `issue_capture` (mission proof)
3. `grounded_report` (report quality)
4. `penalties` (behavior noise)
5. `process_reward` (should be small, never dominant)

If the agent plateaus near 0.2, it is usually living in shaping mode and not finishing with valid submission.

## 10) Tested Behaviors

Current tests verify:

- wrong sensor cannot satisfy thermal coverage,
- thermal-only issue evidence gives partial credit,
- thermal + RGB context gives full issue credit,
- fake IDs and no-evidence reports trigger integrity caps,
- contradictory safety notes reduce integrity,
- low-value duplicates are penalized while useful recaptures are not,
- repeated harmless calls cannot farm process reward,
- both legacy and structured report payloads can complete the mission when properly grounded.
