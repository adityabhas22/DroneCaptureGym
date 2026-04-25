"""Deterministic reward verifiers for SolarInspect evidence."""

from __future__ import annotations

from typing import Any

from dronecaptureops.core.constants import GOOD_CAPTURE_THRESHOLD, MIN_BATTERY_TO_RETURN_PCT
from dronecaptureops.core.models import Capture, EvidenceReport, HiddenDefect, SensorType
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.utils.math_utils import clamp


MIN_ROW_QUALITY = 0.55


def report_cited_photo_ids(report: EvidenceReport | None) -> set[str]:
    """Return every photo ID cited by either compatible or structured report fields."""

    if report is None:
        return set()
    cited = set(report.photo_ids)
    for item in report.evidence:
        cited.update(_ids_from(item, "photo_ids"))
        cited.update(_ids_from(item, "evidence_photo_ids"))
    for item in report.issues_found:
        cited.update(_ids_from(item, "photo_ids"))
        cited.update(_ids_from(item, "evidence_photo_ids"))
    for finding in report.findings:
        cited.update(_ids_from(finding, "photo_ids"))
        cited.update(_ids_from(finding, "evidence_photo_ids"))
    return {photo_id for photo_id in cited if isinstance(photo_id, str)}


def is_photo_valid(
    world: EpisodeWorld,
    photo_id: str,
    *,
    target_id: str | None = None,
    sensor: SensorType | None = None,
    min_quality: float = MIN_ROW_QUALITY,
) -> bool:
    capture = capture_by_id(world, photo_id)
    if capture is None:
        return False
    if sensor is not None and capture.sensor != sensor:
        return False
    if target_id is not None and target_id not in capture.targets_visible:
        return False
    return capture.quality_score >= min_quality


def capture_by_id(world: EpisodeWorld, photo_id: str) -> Capture | None:
    return next((capture for capture in world.capture_log if capture.photo_id == photo_id), None)


def valid_target_captures(
    world: EpisodeWorld,
    target_id: str,
    *,
    sensor: SensorType,
    min_quality: float = MIN_ROW_QUALITY,
    cited_only: bool = False,
) -> list[Capture]:
    cited = report_cited_photo_ids(world.final_report)
    captures = [
        capture
        for capture in world.capture_log
        if capture.sensor == sensor
        and target_id in capture.targets_visible
        and capture.quality_score >= min_quality
        and (not cited_only or capture.photo_id in cited)
    ]
    return captures


def target_visible_in_photo(world: EpisodeWorld, target_id: str, photo_id: str) -> bool:
    capture = capture_by_id(world, photo_id)
    return bool(capture and target_id in capture.targets_visible)


def compute_capture_quality(world: EpisodeWorld, target_id: str, photo_id: str) -> float:
    capture = capture_by_id(world, photo_id)
    if capture is None or target_id not in capture.targets_visible:
        return 0.0
    return capture.quality_score


def compute_required_coverage(world: EpisodeWorld) -> tuple[float, dict[str, Any]]:
    required = list(world.mission.required_rows)
    if not required:
        return 1.0, {"targets_required": 0, "targets_covered": 0}
    covered = [
        row_id
        for row_id in required
        if valid_target_captures(world, row_id, sensor="thermal", min_quality=MIN_ROW_QUALITY)
    ]
    return round(len(covered) / len(required), 4), {
        "targets_required": len(required),
        "targets_covered": len(covered),
        "covered_rows": covered,
        "missing_rows": [row_id for row_id in required if row_id not in covered],
    }


def defect_captured(world: EpisodeWorld, defect: HiddenDefect) -> tuple[float, dict[str, Any]]:
    thermal = [
        capture
        for capture in valid_target_captures(world, defect.target_id, sensor=defect.required_sensor, min_quality=defect.min_quality)
        if defect.defect_id in capture.detected_anomalies
        and capture.resolution_score >= defect.min_resolution_score
        and capture.occlusion_pct <= defect.max_occlusion
        and _view_angle_deg(capture) <= defect.max_view_angle_deg
    ]
    if not thermal:
        return 0.0, {"defect_id": defect.defect_id, "thermal": False, "rgb_context": False}
    if not defect.requires_rgb_context:
        return 1.0, {"defect_id": defect.defect_id, "thermal": True, "rgb_context": True}
    rgb_context = valid_target_captures(world, defect.target_id, sensor="rgb", min_quality=MIN_ROW_QUALITY)
    if rgb_context:
        return 1.0, {"defect_id": defect.defect_id, "thermal": True, "rgb_context": True}
    return 0.6, {"defect_id": defect.defect_id, "thermal": True, "rgb_context": False}


def compute_issue_capture(world: EpisodeWorld) -> tuple[float, dict[str, Any]]:
    defects = list(world.hidden_defects)
    if not defects:
        return 1.0, {"issues_required": 0, "issues_captured": 0}
    weighted = 0.0
    total_weight = 0.0
    details = []
    for defect in defects:
        score, detail = defect_captured(world, defect)
        weight = max(defect.weight, 0.0)
        weighted += weight * score
        total_weight += weight
        details.append({
            "issue_index": len(details),
            "thermal": detail["thermal"],
            "rgb_context": detail["rgb_context"],
            "score": score,
            "weight": weight,
        })
    captured = sum(1 for detail in details if detail["score"] >= 1.0)
    return round(weighted / max(total_weight, 1.0), 4), {
        "issues_required": len(defects),
        "issues_captured": captured,
        "issue_details": details,
    }


def requirement_satisfied(world: EpisodeWorld, requirement_id: str) -> bool:
    if requirement_id.startswith("thermal_overview"):
        coverage, _ = compute_required_coverage(world)
        return coverage >= 1.0
    if requirement_id.startswith("rgb_context_for_"):
        defect_id = requirement_id.removeprefix("rgb_context_for_")
        defect = next((item for item in world.hidden_defects if item.defect_id == defect_id), None)
        if defect is None:
            return False
        score, _ = defect_captured(world, defect)
        return score >= 1.0
    return False


def compute_evidence_success(world: EpisodeWorld, required_coverage: float, issue_capture: float) -> float:
    terminal_safety = 1.0 if (
        (not world.mission.must_return_home or world.checklist_status.returned_home)
        and world.telemetry.battery.level_pct >= world.mission.min_battery_at_done_pct
    ) else 0.0
    # Coverage is the high-weight structured requirement for the MVP; issue and terminal safety close the loop.
    return round(0.55 * required_coverage + 0.35 * issue_capture + 0.10 * terminal_safety, 4)


def compute_operational_efficiency(world: EpisodeWorld, evidence_success: float) -> float:
    completion_gate = min(1.0, evidence_success / 0.85)
    reference_distance = 95.0 + 18.0 * max(len(world.hidden_defects), 1)
    reference_photo_count = 3.0 + 2.0 * len(world.hidden_defects)
    reference_time = 140.0 + 25.0 * len(world.hidden_defects)
    expected_battery_used = 12.0 + 3.0 * len(world.hidden_defects)

    extra_distance = max(0.0, world.distance_flown_m - reference_distance) / max(reference_distance, 1.0)
    extra_photos = max(0.0, len(world.capture_log) - reference_photo_count) / max(reference_photo_count, 1.0)
    extra_time = max(0.0, world.elapsed_time_s - reference_time) / max(reference_time, 1.0)
    battery_used = 100.0 - world.telemetry.battery.level_pct
    battery_overuse = max(0.0, battery_used - expected_battery_used) / max(expected_battery_used, 1.0)

    score = completion_gate * (
        1.0
        - 0.30 * extra_distance
        - 0.25 * extra_photos
        - 0.20 * extra_time
        - 0.25 * battery_overuse
    )
    return round(clamp(score, 0.0, 1.0), 4)


def compute_photo_value(world: EpisodeWorld, capture: Capture) -> float:
    new_target_credit = sum(
        1.0
        for target_id in capture.targets_visible
        if target_id in world.mission.required_rows and capture.sensor == "thermal" and capture.quality_score >= MIN_ROW_QUALITY
    ) / max(len(world.mission.required_rows), 1)
    issue_credit = sum(1.0 for defect_id in capture.detected_anomalies) / max(len(world.hidden_defects), 1)
    quality_credit = capture.quality_score if capture.targets_visible else 0.0
    report_usefulness = 1.0 if capture.photo_id in report_cited_photo_ids(world.final_report) else 0.0
    duplicate_count = sum(
        1
        for prior in world.capture_log
        if prior.photo_id != capture.photo_id
        and prior.sensor == capture.sensor
        and set(prior.targets_visible) == set(capture.targets_visible)
    )
    redundancy_penalty = 0.10 if duplicate_count and new_target_credit < 0.05 and issue_credit < 0.05 else 0.0
    return round(clamp(
        0.35 * new_target_credit
        + 0.25 * new_target_credit
        + 0.20 * issue_credit
        + 0.10 * quality_credit
        + 0.10 * report_usefulness
        - redundancy_penalty,
        0.0,
        1.0,
    ), 4)


def compute_value_per_photo(world: EpisodeWorld) -> float:
    if not world.capture_log:
        return 0.0
    return round(sum(compute_photo_value(world, capture) for capture in world.capture_log) / len(world.capture_log), 4)


def compute_safety_gate(world: EpisodeWorld) -> float:
    cap = 1.0
    for violation in world.safety_violations:
        if "collision" in violation:
            cap = min(cap, 0.0)
        elif "no_fly" in violation:
            cap = min(cap, 0.10)
        elif "privacy" in violation:
            cap = min(cap, 0.20)
        elif "battery_exhausted" in violation:
            cap = min(cap, 0.20)
        elif "unsafe_altitude" in violation:
            cap = min(cap, 0.30)
    if world.done and world.termination_reason == "max_steps" and not world.checklist_status.returned_home:
        cap = min(cap, 0.40)
    if world.done and world.mission.must_return_home and not world.checklist_status.returned_home:
        cap = min(cap, 0.60)
    if world.done and world.telemetry.battery.level_pct < MIN_BATTERY_TO_RETURN_PCT and not world.checklist_status.returned_home:
        cap = min(cap, 0.20)
    if world.invalid_action_count >= 3:
        cap = min(cap, 0.70)
    return round(cap, 4)


def compute_integrity_gate(world: EpisodeWorld) -> tuple[float, list[str]]:
    report = world.final_report
    if report is None:
        return 1.0, []
    warnings: list[str] = []
    cap = 1.0
    real_ids = {capture.photo_id for capture in world.capture_log}
    cited = report_cited_photo_ids(report)
    fake = cited - real_ids
    if fake:
        warnings.append(f"fake photo ids cited: {sorted(fake)}")
        cap = min(cap, 0.20)
    if not world.capture_log:
        warnings.append("report submitted without captured evidence")
        cap = min(cap, 0.20)
    if report.evidence:
        for item in report.evidence:
            if str(item.get("status", "")).lower() == "satisfied":
                ids = _ids_from(item, "photo_ids") or _ids_from(item, "evidence_photo_ids")
                if not ids or not any(photo_id in real_ids for photo_id in ids):
                    warnings.append("satisfied requirement lacks valid photo evidence")
                    cap = min(cap, 0.40)
                if "thermal" in str(item.get("requirement_id", "")).lower():
                    if any(capture_by_id(world, photo_id) and capture_by_id(world, photo_id).sensor != "thermal" for photo_id in ids):
                        warnings.append("thermal requirement cites wrong sensor type")
                        cap = min(cap, 0.50)
                if any(capture_by_id(world, photo_id) and capture_by_id(world, photo_id).quality_score < MIN_ROW_QUALITY for photo_id in ids):
                    warnings.append("satisfied requirement cites low-quality image")
                    cap = min(cap, 0.60)
    known_defects = {defect.defect_id for defect in world.hidden_defects}
    reported_issues = _reported_issue_ids(report)
    unsupported = reported_issues - known_defects
    if unsupported:
        warnings.append(f"unsupported issue claims: {sorted(unsupported)}")
        cap = min(cap, 0.30)
    return round(cap, 4), warnings


def _reported_issue_ids(report: EvidenceReport) -> set[str]:
    ids = {str(item["issue_id"]) for item in report.issues_found if item.get("issue_id")}
    for finding in report.findings:
        for key in ("issue_id", "defect_id", "finding"):
            value = finding.get(key)
            if isinstance(value, str) and value.startswith(("hotspot_", "shadow_")):
                ids.add(value)
    return ids


def _ids_from(item: dict[str, Any], key: str) -> set[str]:
    value = item.get(key)
    if isinstance(value, str):
        return {value}
    if isinstance(value, list):
        return {photo_id for photo_id in value if isinstance(photo_id, str)}
    return set()


def _view_angle_deg(capture: Capture) -> float:
    return (1.0 - clamp(capture.view_angle_score, 0.0, 1.0)) * 90.0
