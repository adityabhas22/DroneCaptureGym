"""Evidence-pack grounding reward."""

from __future__ import annotations

from dronecaptureops.core.models import EvidenceReport
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.base import RewardComponent


def validate_evidence_report(world: EpisodeWorld, report: EvidenceReport | None) -> tuple[float, list[str]]:
    """Validate final report citations against real captures and quality checks."""

    if report is None:
        return 0.0, ["no evidence pack submitted"]
    real_ids = {capture.photo_id for capture in world.capture_log}
    cited_ids = set(report.photo_ids)
    warnings: list[str] = []
    if not cited_ids:
        warnings.append("evidence pack cites no photos")
    fake_ids = cited_ids - real_ids
    if fake_ids:
        warnings.append(f"fake photo ids cited: {sorted(fake_ids)}")

    cited_captures = [capture for capture in world.capture_log if capture.photo_id in cited_ids]
    useful_cited = [
        capture
        for capture in cited_captures
        if capture.targets_visible and capture.quality_score >= world.mission.min_capture_quality
    ]

    required_rows = set(world.mission.required_rows)
    cited_thermal_rows = {
        row_id
        for capture in cited_captures
        if capture.sensor == "thermal" and capture.quality_score >= world.mission.min_capture_quality
        for row_id in capture.targets_visible
        if row_id in required_rows
    }
    missing_row_citations = sorted(required_rows - cited_thermal_rows)
    if missing_row_citations:
        warnings.append(f"missing thermal row citations: {missing_row_citations}")

    thermal_anomalies = set(world.checklist_status.anomalies_detected)
    cited_text = f"{report.summary} {report.findings}".lower()
    missing_anomaly_citations = [anomaly for anomaly in thermal_anomalies if anomaly.lower() not in cited_text]
    if missing_anomaly_citations:
        warnings.append(f"missing anomaly citations: {missing_anomaly_citations}")

    missing_rgb_pairs: list[str] = []
    if world.mission.rgb_closeup_for_anomalies:
        for anomaly, target_id in world.checklist_status.anomaly_targets.items():
            paired_photo_id = world.checklist_status.anomaly_rgb_pairs.get(anomaly)
            paired_capture = next((capture for capture in cited_captures if capture.photo_id == paired_photo_id), None)
            if (
                paired_capture is None
                or paired_capture.sensor != "rgb"
                or target_id not in paired_capture.targets_visible
                or paired_capture.quality_score < world.mission.min_rgb_quality
            ):
                missing_rgb_pairs.append(anomaly)
    if missing_rgb_pairs:
        warnings.append(f"missing rgb anomaly evidence: {missing_rgb_pairs}")

    row_score = len(required_rows & cited_thermal_rows) / max(len(required_rows), 1)
    if not thermal_anomalies:
        anomaly_score = 1.0
    else:
        satisfied = thermal_anomalies - set(missing_anomaly_citations) - set(missing_rgb_pairs)
        anomaly_score = len(satisfied) / len(thermal_anomalies)

    procedure_score = 1.0 if (
        (not world.mission.must_return_home or world.checklist_status.returned_home)
        and (not world.mission.must_return_home or world.checklist_status.landed)
        and world.telemetry.battery.level_pct >= world.mission.min_battery_at_done_pct
    ) else 0.0

    score = 0.0
    if cited_ids and not fake_ids:
        score += 0.20
    if useful_cited:
        score += 0.20
    score += 0.25 * row_score
    score += 0.25 * anomaly_score
    score += 0.10 * procedure_score
    return round(min(score, 1.0), 4), warnings


class ReportGroundingReward(RewardComponent):
    """Rewards evidence packs grounded in real captured photos."""

    name = "report_grounding"

    def compute(self, world: EpisodeWorld) -> float:
        score, _ = validate_evidence_report(world, world.final_report)
        return score
