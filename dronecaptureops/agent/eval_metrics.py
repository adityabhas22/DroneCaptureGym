"""Diagnostic metrics for base-model evaluation.

`success` and `total_reward` alone don't tell us *why* a model failed.
This module extracts the diagnostic profile that does: capability
checkpoints (took off, captured, returned home, ...), tool-use
distribution, failure-mode classification, coverage profile, oracle
comparison, and reward-component decomposition.

The downstream reading: when comparing a 14B vs 32B base, we want to
distinguish "model can't even emit valid tool calls" (format failure)
from "model emits valid calls but submits prematurely" (planning gap)
from "model plans well but flies into the NFZ" (safety gap). Each
maps to a different SFT/PPO fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dronecaptureops.agent.rollout import RolloutResult, RolloutStep


# Failure modes — mutually exclusive, evaluated in order.
FAILURE_MODES = (
    "success",
    "format_collapse",
    "no_takeoff",
    "no_capture",
    "no_submit",
    "premature_submit",
    "coverage_incomplete",
    "anomaly_unconfirmed",
    "didnt_return_home",
    "safety_violation",
    "ran_out_of_steps",
    "submit_rejected",
    "unknown",
)


# Checkpoints — binary milestones along the canonical mission path.
CHECKPOINT_NAMES = (
    "any_valid_action",
    "explored_env",         # called any of get_mission_checklist / list_assets / get_telemetry
    "took_off",
    "captured_thermal",
    "captured_rgb",
    "inspected_capture",
    "covered_any_row",
    "detected_any_anomaly",
    "rgb_paired_anomaly",
    "returned_home",
    "landed",
    "submitted",
    "submission_accepted",
)


@dataclass
class TrajectoryMetrics:
    """The full diagnostic profile of one rollout."""

    checkpoints: dict[str, bool]
    tool_calls: dict[str, int]
    failure_mode: str
    coverage: dict[str, Any]
    safety: dict[str, Any]
    citation_diagnostics: dict[str, Any]
    reward_components: dict[str, float]
    oracle_comparison: dict[str, Any]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def trajectory_metrics(
    result: RolloutResult,
    *,
    oracle_result: RolloutResult | None = None,
) -> TrajectoryMetrics:
    """Compute the full diagnostic profile of a single rollout.

    `oracle_result` is an optional reference trajectory for the same
    (task_id, seed). When provided, oracle-comparison metrics (step
    efficiency, tool-overlap Jaccard) are populated.
    """

    return TrajectoryMetrics(
        checkpoints=extract_checkpoints(result),
        tool_calls=tool_call_distribution(result),
        failure_mode=classify_failure_mode(result),
        coverage=coverage_metrics(result),
        safety=safety_profile(result),
        citation_diagnostics=citation_diagnostics(result),
        reward_components=extract_reward_components(result),
        oracle_comparison=oracle_comparison(result, oracle_result),
    )


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def extract_checkpoints(result: RolloutResult) -> dict[str, bool]:
    """Did the rollout pass each canonical milestone?"""

    final = result.final_observation
    checklist = final.get("checklist_status", {}) or {}
    capture_log = final.get("capture_log", []) or []

    has_thermal_capture = any(c.get("sensor") == "thermal" for c in capture_log)
    has_rgb_capture = any(c.get("sensor") == "rgb" for c in capture_log)

    valid_actions = [step for step in result.trajectory if step.parse_error is None]
    explored_tools = {"get_mission_checklist", "list_assets", "get_telemetry", "get_site_map"}
    explored = any(step.action.get("tool_name") in explored_tools for step in valid_actions)
    took_off = any(step.action.get("tool_name") == "takeoff" for step in valid_actions)
    inspected = any(step.action.get("tool_name") == "inspect_capture" for step in valid_actions)
    submitted = any(step.action.get("tool_name") == "submit_evidence_pack" for step in valid_actions)
    submission_accepted = bool(checklist.get("complete"))

    rows_covered = list(checklist.get("thermal_rows_covered") or [])
    anomalies = list(checklist.get("anomalies_detected") or [])
    rgb_pairs = checklist.get("anomaly_rgb_pairs") or {}

    return {
        "any_valid_action": bool(valid_actions),
        "explored_env": explored,
        "took_off": took_off,
        "captured_thermal": has_thermal_capture,
        "captured_rgb": has_rgb_capture,
        "inspected_capture": inspected,
        "covered_any_row": bool(rows_covered),
        "detected_any_anomaly": bool(anomalies),
        "rgb_paired_anomaly": bool(rgb_pairs),
        "returned_home": bool(checklist.get("returned_home")),
        "landed": bool(checklist.get("landed")),
        "submitted": submitted,
        "submission_accepted": submission_accepted,
    }


def tool_call_distribution(result: RolloutResult) -> dict[str, int]:
    """Count tool-name occurrences across the trajectory.

    Includes parse failures under `_parse_error` so the totals reconcile
    to len(trajectory).
    """

    distribution: dict[str, int] = {}
    for step in result.trajectory:
        if step.parse_error is not None:
            distribution["_parse_error"] = distribution.get("_parse_error", 0) + 1
            continue
        name = step.action.get("tool_name") or "_unknown"
        distribution[name] = distribution.get(name, 0) + 1
    return distribution


def classify_failure_mode(result: RolloutResult) -> str:
    """Single-label classification of why the episode ended where it did.

    Modes are checked in order; the first applicable label wins. Ties are
    broken by listing the lowest-blame mode first (e.g. format_collapse
    is checked before no_takeoff because if the model can't emit valid
    actions, "didn't take off" is downstream).
    """

    final = result.final_observation
    checklist = final.get("checklist_status", {}) or {}
    capture_log = final.get("capture_log", []) or []
    metadata = final.get("metadata", {}) or {}
    breakdown = result.reward_breakdown

    if checklist.get("complete"):
        return "success"

    parse_errors = sum(1 for step in result.trajectory if step.parse_error)
    if result.steps and parse_errors / max(result.steps, 1) >= 0.5:
        return "format_collapse"

    valid_actions = [step for step in result.trajectory if step.parse_error is None]

    took_off = any(step.action.get("tool_name") == "takeoff" for step in valid_actions)
    if not took_off:
        return "no_takeoff"

    if not capture_log:
        return "no_capture"

    submitted = any(step.action.get("tool_name") == "submit_evidence_pack" for step in valid_actions)
    if not submitted:
        # Episode ended without submission — figure out why.
        if metadata.get("termination_reason") == "max_steps":
            return "ran_out_of_steps"
        # Hard safety cap — the env terminates with a 0-cap; check breakdown.
        if breakdown.get("safety_gate", 1.0) <= 0.30:
            return "safety_violation"
        return "no_submit"

    # Submitted but not complete. Diagnose which gate failed.
    if breakdown.get("safety_gate", 1.0) <= 0.30:
        return "safety_violation"
    rows_covered = set(checklist.get("thermal_rows_covered") or [])
    rows_required = set(_required_rows(result))
    missing_rows = rows_required - rows_covered
    anomalies = set(checklist.get("anomalies_detected") or [])
    paired = set(checklist.get("anomaly_rgb_pairs") or {})
    unpaired = anomalies - paired

    submit_action_result = _submit_action_result(result)
    if submit_action_result is not None and submit_action_result.get("accepted") is False:
        # Distinguish "premature with no captures" from "incomplete coverage".
        if not rows_covered:
            return "premature_submit"
        if missing_rows:
            return "coverage_incomplete"
        if unpaired:
            return "anomaly_unconfirmed"
        if not checklist.get("returned_home"):
            return "didnt_return_home"
        return "submit_rejected"

    return "unknown"


def coverage_metrics(result: RolloutResult) -> dict[str, Any]:
    final = result.final_observation
    checklist = final.get("checklist_status", {}) or {}
    rows_required = list(_required_rows(result))
    rows_covered = list(checklist.get("thermal_rows_covered") or [])
    anomalies = list(checklist.get("anomalies_detected") or [])
    paired = checklist.get("anomaly_rgb_pairs") or {}
    return {
        "rows_required": len(rows_required),
        "rows_covered": len(rows_covered),
        "rows_covered_fraction": (len(rows_covered) / len(rows_required)) if rows_required else 0.0,
        "anomalies_detected": len(anomalies),
        "anomalies_rgb_paired": len(paired),
        "rgb_pairing_fraction": (len(paired) / len(anomalies)) if anomalies else 1.0,
        "missing_rows": [row for row in rows_required if row not in rows_covered],
        "unpaired_anomalies": [a for a in anomalies if a not in paired],
    }


def safety_profile(result: RolloutResult) -> dict[str, Any]:
    """Count safety violations by category from observation warnings."""

    final = result.final_observation
    warnings = final.get("warnings", []) or []
    breakdown = result.reward_breakdown
    categories: dict[str, int] = {}
    for warning in warnings:
        if not isinstance(warning, str):
            continue
        for kind in ("no_fly_violation", "obstacle_violation", "privacy_capture_violation",
                     "unsafe_altitude", "unsafe_speed", "battery_exhausted",
                     "invalid_gimbal_pitch", "invalid_gimbal_yaw"):
            if kind in warning:
                categories[kind] = categories.get(kind, 0) + 1
                break
    return {
        "safety_gate": breakdown.get("safety_gate", 1.0),
        "integrity_gate": breakdown.get("integrity_gate", 1.0),
        "violation_categories": categories,
        "total_violations": sum(categories.values()),
    }


def citation_diagnostics(result: RolloutResult) -> dict[str, Any]:
    """Surface report-citation anti-gaming warnings for benchmark aggregation."""

    submit_result = _submit_action_result(result) or {}
    submit_warnings = [
        warning for warning in submit_result.get("warnings", [])
        if isinstance(warning, str)
    ]
    debug = result.reward_breakdown.get("debug") or {}
    integrity_warnings = [
        warning for warning in debug.get("integrity_warnings", [])
        if isinstance(warning, str)
    ]
    warnings = submit_warnings + [warning for warning in integrity_warnings if warning not in submit_warnings]
    return {
        "submit_warnings": submit_warnings,
        "integrity_warnings": integrity_warnings,
        "overbroad_citations": any("overbroad" in warning or "irrelevant photos" in warning for warning in warnings),
        "issue_specific_evidence_missing": any("issue-specific cited evidence" in warning for warning in warnings),
        "hallucinated_issue_claim": any("no-anomaly" in warning or "non-reportable issue" in warning for warning in warnings),
        "fake_or_unsupported_claim": any("fake photo ids" in warning or "unsupported issue claims" in warning for warning in warnings),
    }


def extract_reward_components(result: RolloutResult) -> dict[str, float]:
    """Pull the reward breakdown components we care about into a flat dict."""

    breakdown = result.reward_breakdown
    keys = (
        "total",
        "evidence_success",
        "required_coverage",
        "issue_capture",
        "operational_efficiency",
        "grounded_report",
        "process_reward",
        "integrity_gate",
        "safety_gate",
        "penalties",
        "capture_quality",
        "battery_management",
        "report_grounding",
    )
    return {key: float(breakdown.get(key) or 0.0) for key in keys}


def oracle_comparison(
    result: RolloutResult,
    oracle_result: RolloutResult | None,
) -> dict[str, Any]:
    """Compare a base-model rollout against the oracle reference."""

    if oracle_result is None:
        return {"available": False}

    base_tools = _tool_sequence(result)
    oracle_tools = _tool_sequence(oracle_result)
    base_set = set(base_tools)
    oracle_set = set(oracle_tools)

    union = base_set | oracle_set
    intersection = base_set & oracle_set
    jaccard = (len(intersection) / len(union)) if union else 1.0

    # Multiset overlap — how many oracle calls did the base also make.
    base_counts: dict[str, int] = {}
    for tool in base_tools:
        base_counts[tool] = base_counts.get(tool, 0) + 1
    oracle_counts: dict[str, int] = {}
    for tool in oracle_tools:
        oracle_counts[tool] = oracle_counts.get(tool, 0) + 1
    multiset_overlap = sum(min(base_counts.get(t, 0), oracle_counts.get(t, 0)) for t in oracle_set)
    multiset_recall = (multiset_overlap / sum(oracle_counts.values())) if oracle_counts else 0.0

    return {
        "available": True,
        "oracle_steps": oracle_result.steps,
        "base_steps": result.steps,
        "step_ratio": (result.steps / oracle_result.steps) if oracle_result.steps else 0.0,
        "tool_jaccard": round(jaccard, 4),
        "oracle_tool_recall": round(multiset_recall, 4),
        "tools_oracle_used_base_skipped": sorted(oracle_set - base_set),
        "tools_base_used_oracle_skipped": sorted(base_set - oracle_set),
    }


# ---------------------------------------------------------------------------
# Aggregation across many rows
# ---------------------------------------------------------------------------


def aggregate_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-model breakdowns across many evaluation rows.

    `rows` are dicts (typically deserialized from the eval JSONL). Each
    must include `model`, `task_id`, `failure_mode`, `checkpoints`,
    `tool_calls`.
    """

    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    summary: dict[str, Any] = {}
    for model, model_rows in by_model.items():
        n = len(model_rows)
        failure_counts: dict[str, int] = {}
        checkpoint_counts: dict[str, int] = {name: 0 for name in CHECKPOINT_NAMES}
        tool_totals: dict[str, int] = {}
        for row in model_rows:
            failure_counts[row["failure_mode"]] = failure_counts.get(row["failure_mode"], 0) + 1
            for name in CHECKPOINT_NAMES:
                if row["checkpoints"].get(name):
                    checkpoint_counts[name] += 1
            for tool, count in (row.get("tool_calls") or {}).items():
                tool_totals[tool] = tool_totals.get(tool, 0) + count
        summary[model] = {
            "n": n,
            "failure_mode_distribution": {
                mode: round(failure_counts.get(mode, 0) / n, 4) for mode in FAILURE_MODES if failure_counts.get(mode)
            },
            "checkpoint_completion_rate": {
                name: round(checkpoint_counts[name] / n, 4) for name in CHECKPOINT_NAMES
            },
            "tool_calls_per_episode": {
                tool: round(count / n, 3) for tool, count in sorted(tool_totals.items(), key=lambda kv: -kv[1])
            },
            "citation_diagnostic_rate": _citation_diagnostic_rates(model_rows),
        }
    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _required_rows(result: RolloutResult) -> tuple[str, ...]:
    mission = result.initial_observation.get("mission") or {}
    return tuple(mission.get("required_rows") or ())


def _submit_action_result(result: RolloutResult) -> dict[str, Any] | None:
    for step in reversed(result.trajectory):
        if step.action.get("tool_name") == "submit_evidence_pack":
            return step.action_result
    return None


def _tool_sequence(result: RolloutResult) -> list[str]:
    return [
        step.action.get("tool_name", "_unknown")
        for step in result.trajectory
        if step.parse_error is None
    ]


def _citation_diagnostic_rates(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "overbroad_citations",
        "issue_specific_evidence_missing",
        "hallucinated_issue_claim",
        "fake_or_unsupported_claim",
    )
    n = max(len(rows), 1)
    rates: dict[str, float] = {}
    for key in keys:
        rates[key] = round(sum(1 for row in rows if (row.get("citation_diagnostics") or {}).get(key)) / n, 4)
    return rates


__all__ = [
    "CHECKPOINT_NAMES",
    "FAILURE_MODES",
    "TrajectoryMetrics",
    "aggregate_diagnostics",
    "classify_failure_mode",
    "citation_diagnostics",
    "coverage_metrics",
    "extract_checkpoints",
    "extract_reward_components",
    "oracle_comparison",
    "safety_profile",
    "tool_call_distribution",
    "trajectory_metrics",
]
