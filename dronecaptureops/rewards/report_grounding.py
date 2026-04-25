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
    useful_cited = [
        capture for capture in world.capture_log
        if capture.photo_id in cited_ids and capture.targets_visible and capture.quality_score >= 0.55
    ]
    thermal_anomalies = {
        anomaly
        for capture in world.capture_log
        if capture.sensor == "thermal"
        for anomaly in capture.detected_anomalies
    }
    cited_text = f"{report.summary} {report.findings}".lower()
    missing_anomaly_citations = [anomaly for anomaly in thermal_anomalies if anomaly.lower() not in cited_text]
    if missing_anomaly_citations:
        warnings.append(f"missing anomaly citations: {missing_anomaly_citations}")
    score = 0.0
    if cited_ids and not fake_ids:
        score += 0.35
    if useful_cited:
        score += 0.35
    if thermal_anomalies and not missing_anomaly_citations:
        score += 0.20
    if world.checklist_status.complete:
        score += 0.10
    return round(min(score, 1.0), 4), warnings


class ReportGroundingReward(RewardComponent):
    """Rewards evidence packs grounded in real captured photos."""

    name = "report_grounding"

    def compute(self, world: EpisodeWorld) -> float:
        score, _ = validate_evidence_report(world, world.final_report)
        return score
