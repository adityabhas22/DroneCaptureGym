"""Evidence-pack grounding reward."""

from __future__ import annotations

from dronecaptureops.core.models import EvidenceReport
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.base import RewardComponent
from dronecaptureops.rewards.verifiers import (
    MIN_ROW_QUALITY,
    compute_integrity_gate,
    report_cited_photo_ids,
)


def validate_evidence_report(world: EpisodeWorld, report: EvidenceReport | None) -> tuple[float, list[str]]:
    """Validate final report citations against real captures and quality checks."""

    if report is None:
        return 0.0, ["no evidence pack submitted"]
    real_ids = {capture.photo_id for capture in world.capture_log}
    cited_ids = report_cited_photo_ids(report)
    warnings: list[str] = []
    integrity_cap, integrity_warnings = compute_integrity_gate(world)
    warnings.extend(integrity_warnings)
    if not cited_ids:
        warnings.append("evidence pack cites no photos")
    fake_ids = cited_ids - real_ids
    if fake_ids:
        warnings.append(f"fake photo ids cited: {sorted(fake_ids)}")
    useful_cited = [
        capture for capture in world.capture_log
        if capture.photo_id in cited_ids and capture.targets_visible and capture.quality_score >= MIN_ROW_QUALITY
    ]
    thermal_anomalies = {
        anomaly
        for capture in world.capture_log
        if capture.sensor == "thermal"
        for anomaly in capture.detected_anomalies
    }
    cited_text = f"{report.summary} {report.findings} {report.evidence} {report.issues_found}".lower()
    missing_anomaly_citations = [anomaly for anomaly in thermal_anomalies if anomaly.lower() not in cited_text]
    if missing_anomaly_citations:
        warnings.append(f"missing anomaly citations: {missing_anomaly_citations}")

    requirement_linking = 1.0 if cited_ids and not fake_ids and useful_cited else 0.0
    if report.evidence:
        satisfied_items = [item for item in report.evidence if str(item.get("status", "")).lower() == "satisfied"]
        linked_items = [
            item
            for item in satisfied_items
            if set(item.get("photo_ids", []) or item.get("evidence_photo_ids", [])) & real_ids
        ]
        requirement_linking = len(linked_items) / max(len(satisfied_items), 1)

    if not thermal_anomalies:
        issue_linking = 1.0
    else:
        issue_linking = 1.0 if not missing_anomaly_citations and useful_cited else 0.0
        if report.issues_found:
            linked_issues = [
                issue
                for issue in report.issues_found
                if set(issue.get("evidence_photo_ids", []) or issue.get("photo_ids", [])) & real_ids
            ]
            issue_linking = len(linked_issues) / max(len(report.issues_found), 1)

    missing_rows = set(world.mission.required_rows) - set(world.checklist_status.thermal_rows_covered)
    open_item_accuracy = 1.0 if not missing_rows else float(bool(report.open_items))
    if missing_rows and not report.open_items:
        warnings.append("missing open items for incomplete coverage")

    safety_note_text = " ".join(report.safety_notes).lower()
    safety_note_accuracy = 1.0
    if report.safety_notes:
        mentions_return = "return" in safety_note_text or "home" in safety_note_text
        mentions_battery = "battery" in safety_note_text or "%" in safety_note_text
        safety_note_accuracy = 1.0 if mentions_return and mentions_battery else 0.5
    elif world.done:
        safety_note_accuracy = 0.0
        warnings.append("missing safety note")

    score = (
        0.40 * requirement_linking
        + 0.25 * issue_linking
        + 0.20 * open_item_accuracy
        + 0.15 * safety_note_accuracy
    )
    return round(min(score, integrity_cap, 1.0), 4), warnings


class ReportGroundingReward(RewardComponent):
    """Rewards evidence packs grounded in real captured photos."""

    name = "report_grounding"

    def compute(self, world: EpisodeWorld) -> float:
        score, _ = validate_evidence_report(world, world.final_report)
        return score
