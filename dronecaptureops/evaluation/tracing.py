"""Trajectory trace artifacts for debugging policy behavior and rewards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from dronecaptureops.evaluation.rollout import RolloutResult, RolloutStep


class StateChange(BaseModel):
    """A compact visible state change."""

    path: str
    before: Any = None
    after: Any = None
    summary: str


class TraceStep(BaseModel):
    """One human-debuggable step."""

    step: int
    action: dict[str, Any]
    reward: float
    reward_breakdown: dict[str, Any]
    reward_delta: dict[str, float]
    state_changes: list[StateChange] = Field(default_factory=list)
    action_result: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    done: bool = False


class EpisodeTrace(BaseModel):
    """Trace view of a rollout."""

    policy_name: str
    seed: int | None
    scenario_family: str | None
    episode_id: str | None
    initial_summary: dict[str, Any]
    steps: list[TraceStep]
    final_reward_breakdown: dict[str, Any]
    final_observation: dict[str, Any]
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            "# DroneCaptureOps Trace",
            "",
            f"- policy: `{self.policy_name}`",
            f"- scenario_family: `{self.scenario_family}`",
            f"- seed: `{self.seed}`",
            f"- episode_id: `{self.episode_id}`",
            f"- final_reward: `{_fmt(self.final_reward_breakdown.get('total', 0.0))}`",
            "",
        ]
        if self.diagnostics:
            lines.append("## Why This Episode Scored What It Scored")
            lines.append("")
            for key in (
                "submitted",
                "accepted",
                "captured_required_coverage",
                "cited_required_coverage",
                "cited_issue_capture",
                "fake_photo_ids_cited",
                "low_quality_cited_photo_ids",
                "missing_cited_rows",
                "integrity_warnings",
                "submit_warnings",
            ):
                if key in self.diagnostics:
                    lines.append(f"- {key}: `{_compact(self.diagnostics[key])}`")
            lines.append("")
        lines.append("## Initial Summary")
        for key, value in self.initial_summary.items():
            lines.append(f"- {key}: `{_compact(value)}`")
        for step in self.steps:
            tool = step.action.get("tool_name", "unknown")
            lines.extend(["", f"## Step {step.step}: `{tool}`", "", "```json", json.dumps(step.action, indent=2, sort_keys=True), "```"])
            lines.append(f"\nReward: `{_fmt(step.reward)}`")
            lines.extend(["", "**Reward Deltas**"])
            if step.reward_delta:
                for component, delta in sorted(step.reward_delta.items()):
                    sign = "+" if delta >= 0 else ""
                    lines.append(f"- {component}: {sign}{_fmt(delta)}")
            else:
                lines.append("- no reward component changes")
            lines.extend(["", "**State Changes**"])
            if step.state_changes:
                for change in step.state_changes:
                    lines.append(f"- {change.summary}")
            else:
                lines.append("- no material visible state changes")
            if step.warnings:
                lines.extend(["", "**Warnings**"])
                lines.extend(f"- {warning}" for warning in step.warnings)
        return "\n".join(lines) + "\n"


def trace_rollout(rollout: RolloutResult | dict[str, Any]) -> EpisodeTrace:
    """Build a trace with reward deltas and visible state changes."""

    result = rollout if isinstance(rollout, RolloutResult) else RolloutResult.model_validate(rollout)
    return EpisodeTrace(
        policy_name=result.policy_name,
        seed=result.seed,
        scenario_family=result.scenario_family,
        episode_id=result.episode_id,
        initial_summary=_initial_summary(result.initial_observation),
        steps=[_trace_step(step) for step in result.trajectory],
        final_reward_breakdown=result.reward_breakdown,
        final_observation=result.final_observation,
        diagnostics=_final_report_diagnostics(result),
    )


def build_trace_artifacts(rollout: RolloutResult | dict[str, Any]) -> dict[str, Any]:
    """Return all JSON artifacts needed for benchmark/trajectory debugging."""

    result = rollout if isinstance(rollout, RolloutResult) else RolloutResult.model_validate(rollout)
    trace = trace_rollout(result)
    return {
        "episode_steps": [step.model_dump(mode="json") for step in result.trajectory],
        "evidence_log": _evidence_log(result),
        "route_log": _route_log(result),
        "inspection_report": _inspection_report(result),
        "capture_table": _capture_table(result),
        "reward_evolution": _reward_evolution(result),
        "safety_timeline": _safety_timeline(result),
        "final_report_diagnostics": _final_report_diagnostics(result),
        "trace": trace.model_dump(mode="json"),
        "trace_markdown": trace.to_markdown(),
    }


def write_trace_artifacts(rollout: RolloutResult | dict[str, Any], output_dir: str | Path) -> dict[str, Path]:
    """Write JSON and Markdown trace artifacts."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifacts = build_trace_artifacts(rollout)
    paths = {
        "episode_steps": out / "episode_steps.json",
        "evidence_log": out / "evidence_log.json",
        "route_log": out / "route_log.json",
        "inspection_report": out / "inspection_report.json",
        "capture_table": out / "capture_table.json",
        "reward_evolution": out / "reward_evolution.json",
        "safety_timeline": out / "safety_timeline.json",
        "final_report_diagnostics": out / "final_report_diagnostics.json",
        "trace": out / "trace.json",
        "trace_markdown": out / "trace.md",
    }
    for key, path in paths.items():
        if key == "trace_markdown":
            path.write_text(str(artifacts[key]), encoding="utf-8")
        else:
            path.write_text(json.dumps(artifacts[key], indent=2, sort_keys=True), encoding="utf-8")
    return paths


def _capture_table(result: RolloutResult) -> list[dict[str, Any]]:
    """Compact per-photo summary for fast eyeballing of an episode's evidence."""

    final = result.final_observation
    captures = final.get("capture_log") or final.get("evidence_artifacts") or []
    cited = _cited_photo_ids_from_observation(final)
    rows = []
    for capture in captures:
        photo_id = capture.get("photo_id")
        rows.append({
            "photo_id": photo_id,
            "sensor": capture.get("sensor"),
            "label": capture.get("label"),
            "targets_visible": capture.get("targets_visible") or [],
            "per_target_quality": capture.get("per_target_quality") or {},
            "quality_score": capture.get("quality_score"),
            "detected_anomalies": capture.get("detected_anomalies") or [],
            "occlusion_pct": capture.get("occlusion_pct"),
            "cited_in_report": photo_id in cited,
        })
    return rows


def _reward_evolution(result: RolloutResult) -> list[dict[str, Any]]:
    """Per-step total + every numeric breakdown component."""

    rows = []
    for step in result.trajectory:
        components = {
            key: float(value)
            for key, value in step.reward_breakdown.items()
            if isinstance(value, int | float)
        }
        rows.append({
            "step": step.step,
            "tool_name": step.action.get("tool_name"),
            "reward": step.reward,
            "components": components,
            "deltas": dict(step.reward_delta),
        })
    return rows


def _safety_timeline(result: RolloutResult) -> list[dict[str, Any]]:
    """One row per step where safety, integrity, or warnings change."""

    rows = []
    prior_violations: list[str] = []
    prior_integrity_warnings: list[str] = []
    for step in result.trajectory:
        breakdown = step.reward_breakdown
        debug = breakdown.get("debug") if isinstance(breakdown, dict) else None
        integrity_warnings = (debug or {}).get("integrity_warnings") or []
        observation = step.next_observation
        violations = _safety_violations_from_observation(observation)
        new_violations = [v for v in violations if v not in prior_violations]
        new_integrity = [w for w in integrity_warnings if w not in prior_integrity_warnings]
        if new_violations or new_integrity or step.warnings:
            rows.append({
                "step": step.step,
                "tool_name": step.action.get("tool_name"),
                "new_safety_violations": new_violations,
                "new_integrity_warnings": new_integrity,
                "step_warnings": list(step.warnings),
                "safety_gate": breakdown.get("safety_gate") if isinstance(breakdown, dict) else None,
                "integrity_gate": breakdown.get("integrity_gate") if isinstance(breakdown, dict) else None,
            })
        prior_violations = list(violations)
        prior_integrity_warnings = list(integrity_warnings)
    return rows


def _final_report_diagnostics(result: RolloutResult) -> dict[str, Any]:
    """Why the final report scored what it scored — missing rows, fake IDs, etc."""

    final = result.final_observation
    breakdown = final.get("reward_breakdown") or {}
    debug = breakdown.get("debug") or {}
    cited = _cited_photo_ids_from_observation(final)
    captures = final.get("capture_log") or final.get("evidence_artifacts") or []
    real_ids = {capture.get("photo_id") for capture in captures}
    fake_ids = sorted(cited - real_ids)
    cited_captures = [capture for capture in captures if capture.get("photo_id") in cited]
    low_quality_cited = [
        capture.get("photo_id")
        for capture in cited_captures
        if isinstance(capture.get("quality_score"), int | float)
        and capture["quality_score"] < 0.55
    ]
    submit_steps = [
        step for step in result.trajectory
        if step.action.get("tool_name") == "submit_evidence_pack"
    ]
    submitted = bool(submit_steps)
    last_action_result = submit_steps[-1].action_result if submit_steps else {}
    return {
        "submitted": submitted,
        "accepted": last_action_result.get("accepted"),
        "submit_warnings": last_action_result.get("warnings") or [],
        "missing_cited_rows": debug.get("missing_cited_rows") or [],
        "captured_required_coverage": debug.get("captured_required_coverage"),
        "cited_required_coverage": debug.get("cited_required_coverage"),
        "captured_issue_capture": debug.get("captured_issue_capture"),
        "cited_issue_capture": debug.get("cited_issue_capture"),
        "integrity_warnings": debug.get("integrity_warnings") or [],
        "fake_photo_ids_cited": fake_ids,
        "low_quality_cited_photo_ids": low_quality_cited,
        "cited_photo_count": len(cited),
        "useful_cited_photo_count": len(cited_captures) - len(low_quality_cited),
        "safety_gate": breakdown.get("safety_gate"),
        "integrity_gate": breakdown.get("integrity_gate"),
        "raw_outcome_if_submitted": debug.get("raw_outcome_if_submitted"),
    }


def _cited_photo_ids_from_observation(observation: dict[str, Any]) -> set[str]:
    """Pull cited photo IDs from a final observation's submit action_result, if any."""

    cited: set[str] = set()
    action_result = observation.get("action_result") or {}
    report = action_result.get("evidence") or action_result.get("report") or {}
    for key in ("photo_ids", "evidence_photo_ids"):
        for value in (report.get(key) if isinstance(report, dict) else None) or []:
            if isinstance(value, str):
                cited.add(value)
    # Capture log may carry a direct cited-flag if present.
    for capture in observation.get("capture_log") or []:
        if capture.get("cited") is True and isinstance(capture.get("photo_id"), str):
            cited.add(capture["photo_id"])
    return cited


def _safety_violations_from_observation(observation: dict[str, Any]) -> list[str]:
    """Extract safety violations from an observation's warnings/state_summary."""

    violations: list[str] = []
    for warning in observation.get("warnings") or []:
        if isinstance(warning, str) and ("violation" in warning or "unsafe" in warning or "no_fly" in warning):
            violations.append(warning)
    return violations


def _trace_step(step: RolloutStep) -> TraceStep:
    return TraceStep(
        step=step.step,
        action=step.action,
        reward=step.reward,
        reward_breakdown=step.reward_breakdown,
        reward_delta=step.reward_delta,
        state_changes=_state_changes(step.observation, step.next_observation),
        action_result=step.action_result,
        warnings=step.warnings,
        done=step.done,
    )


def _initial_summary(observation: dict[str, Any]) -> dict[str, Any]:
    mission = observation.get("mission") or {}
    telemetry = observation.get("telemetry") or {}
    autopilot = telemetry.get("autopilot") or {}
    affordances = observation.get("inspection_affordances") or {}
    return {
        "mission_id": mission.get("mission_id"),
        "scenario_family": mission.get("scenario_family"),
        "difficulty": mission.get("difficulty"),
        "mode": autopilot.get("mode"),
        "battery_pct": (telemetry.get("battery") or {}).get("level_pct"),
        "mission_phase": affordances.get("mission_phase"),
        "pending_assets": affordances.get("pending_asset_ids") or [],
    }


def _state_changes(before: dict[str, Any], after: dict[str, Any]) -> list[StateChange]:
    changes: list[StateChange] = []
    paths = {
        "telemetry.autopilot.mode": (_nested(before, "telemetry", "autopilot", "mode"), _nested(after, "telemetry", "autopilot", "mode")),
        "telemetry.pose": (_nested(before, "telemetry", "pose"), _nested(after, "telemetry", "pose")),
        "telemetry.battery.level_pct": (_nested(before, "telemetry", "battery", "level_pct"), _nested(after, "telemetry", "battery", "level_pct")),
        "telemetry.gimbal.target_asset_id": (_nested(before, "telemetry", "gimbal", "target_asset_id"), _nested(after, "telemetry", "gimbal", "target_asset_id")),
        "inspection_affordances.mission_phase": (_nested(before, "inspection_affordances", "mission_phase"), _nested(after, "inspection_affordances", "mission_phase")),
        "checklist_status": (before.get("checklist_status"), after.get("checklist_status")),
        "evidence_artifacts": (len(before.get("evidence_artifacts") or []), len(after.get("evidence_artifacts") or [])),
        "warnings": (before.get("warnings") or [], after.get("warnings") or []),
    }
    for path, (old, new) in paths.items():
        if old != new:
            changes.append(StateChange(path=path, before=old, after=new, summary=f"{path}: {_compact(old)} -> {_compact(new)}"))
    return changes


def _route_log(result: RolloutResult) -> list[dict[str, Any]]:
    rows = []
    for step in result.trajectory:
        telemetry = step.next_observation.get("telemetry") or {}
        rows.append(
            {
                "step": step.step,
                "tool_name": step.action.get("tool_name"),
                "pose": telemetry.get("pose"),
                "mode": (telemetry.get("autopilot") or {}).get("mode"),
                "battery_pct": (telemetry.get("battery") or {}).get("level_pct"),
                "distance_flown_m": telemetry.get("distance_flown_m"),
                "elapsed_time_s": telemetry.get("elapsed_time_s"),
            }
        )
    return rows


def _evidence_log(result: RolloutResult) -> list[dict[str, Any]]:
    final = result.final_observation
    return list(final.get("evidence_artifacts") or final.get("capture_log") or [])


def _inspection_report(result: RolloutResult) -> dict[str, Any]:
    submit_steps = [
        step for step in result.trajectory
        if step.action.get("tool_name") == "submit_evidence_pack"
    ]
    if not submit_steps:
        return {"submitted": False, "checklist_status": result.final_observation.get("checklist_status")}
    last = submit_steps[-1]
    return {
        "submitted": True,
        "action_arguments": last.action.get("arguments", {}),
        "action_result": last.action_result,
        "checklist_status": result.final_observation.get("checklist_status"),
    }


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _compact(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 120 else text[:117] + "..."


def _fmt(value: Any) -> str:
    return f"{float(value):.4f}" if isinstance(value, int | float) else str(value)
