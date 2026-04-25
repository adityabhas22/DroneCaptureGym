# SolarInspect Rewards Implementation

This branch implements the Solar Farm Inspection reward specification on top of the richer environment model that is now on `main`. The goal is to reward complete, safe, grounded inspection evidence rather than waypoint visiting or photo count.

## What Changed

Terminal mission scoring now follows the outcome-first formula from the specification:

```text
R_total =
  min(
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

The old reward fields are still populated for compatibility:

- `target_coverage` mirrors `required_coverage`
- `defect_visibility` mirrors `issue_capture`
- `route_efficiency` mirrors `operational_efficiency`
- `report_grounding` mirrors `grounded_report`

New logged reward fields include:

- `evidence_success`
- `required_coverage`
- `issue_capture`
- `operational_efficiency`
- `grounded_report`
- `process_reward`
- `integrity_gate`
- `value_per_photo`
- `debug`

Before final report submission, the environment still emits learning signal, but that value is treated as capped shaping/progress rather than the actual mission score. The debug payload makes this explicit:

```json
{
  "terminal_submitted": false,
  "nonterminal_cap_applied": true,
  "shaping_reward": 0.14,
  "raw_outcome_if_submitted": 0.72
}
```

After `submit_evidence_pack`, `total` is the terminal mission score after safety and integrity caps.

## Verifier Utilities

The new `dronecaptureops.rewards.verifiers` module centralizes deterministic checks used by the reward system:

- photo ID validity
- target visibility in a photo
- capture quality for a target/photo pair
- required thermal row coverage
- hidden issue capture
- evidence success
- operational efficiency
- safety cap calculation
- integrity cap calculation
- value per photo

Verifier scoring now separates captured progress from cited terminal proof. Captured thermal coverage and issue visibility are useful during exploration, while terminal `evidence_success` is based on final report citations. A single valid cited thermal photo can satisfy multiple required rows, but every required row must be visible in at least one cited valid thermal photo.

These checks use simulator state and capture metadata, not an LLM judge.

## Safety and Integrity Gates

Safety is now cap-based rather than a binary zero. For example, a no-fly violation caps total reward at `0.10`, while timeout away from home caps at `0.40`. This keeps a learning signal for recoverable failures while preventing unsafe missions from scoring highly.

The integrity gate caps rewards for evidence hallucination and report misuse, including:

- fake photo IDs
- reports submitted without captured evidence
- satisfied requirements without valid evidence
- wrong sensor citations
- low-quality evidence cited as definitive
- unsupported issue claims
- safety notes that contradict telemetry
- structured `satisfied` claims that do not prove the stated requirement

## Evidence Reports

`submit_evidence_pack` now supports both the existing payload shape and the structured report shape from the specification.

Existing compatible payload:

```json
{
  "summary": "Rows B4-B8 inspected.",
  "photo_ids": ["IMG-T-001", "IMG-R-002"],
  "findings": [{"finding": "hotspot_B6", "photo_ids": ["IMG-T-001", "IMG-R-002"]}]
}
```

Structured payload:

```json
{
  "mission_status": "complete",
  "evidence": [
    {
      "requirement_id": "thermal_overview_rows_B4_B8",
      "status": "satisfied",
      "photo_ids": ["IMG-T-001"]
    }
  ],
  "issues_found": [
    {
      "issue_id": "hotspot_B6",
      "evidence_photo_ids": ["IMG-T-001", "IMG-R-002"],
      "recommended_followup": "manual review"
    }
  ],
  "open_items": [],
  "safety_notes": ["Returned home with battery reserve."]
}
```

## Process Rewards and Penalties

Process reward is intentionally small and capped at `0.10`. It rewards useful learning signals such as valid tool calls, accepted safe flight, useful captures, correct sensor use, improved recaptures, return-home behavior, and inspecting captures before reporting.

Penalties cover invalid actions, redundant low-value captures, premature/no-evidence reports, and missions that end without returning home when required.

## Capture and Report Behavior

RGB anomaly follow-up is stricter now: an RGB capture only pairs with a detected anomaly if it actually sees that anomaly's target row with sufficient quality. This prevents a generic RGB photo from satisfying every detected issue.

Report grounding checks all cited photo IDs across both compatible and structured report fields. Thermal requirements must cite thermal evidence, and issue reports must cite real captured evidence.

For defects that require RGB context, full terminal issue credit requires cited valid thermal defect evidence and cited valid RGB context evidence for the defect target. Thermal-only defect evidence still gives partial exploration credit, but it cannot become full terminal issue credit.

Redundancy penalties are based on low marginal value rather than repeated targets alone. Useful recaptures, better-quality images, RGB context, issue confirmations, and cited evidence are not penalized simply because they overlap with earlier photos.

## Tests Added or Updated

The reward tests now cover:

- new reward breakdown shape
- no-fly safety cap behavior
- fake photo ID integrity cap
- report without evidence cap
- wrong sensor not satisfying thermal coverage
- thermal-only anomaly partial credit
- thermal plus RGB anomaly full credit
- redundant photo penalty
- early submission scoring low
- backward-compatible evidence reports
- structured evidence reports
- nonterminal shaping logs
- process-reward farming resistance
- cited terminal coverage
- telemetry-contradicting safety notes
- low-value versus useful redundancy

Current verification:

```text
python -m pytest
20 passed

python examples\run_scripted_agent.py
reward: 1.0
done: true
```
