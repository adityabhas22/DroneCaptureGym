"""Evidence-pack tool handlers."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from dronecaptureops.core.models import EvidenceReport
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.report_grounding import validate_evidence_report
from dronecaptureops.rewards.verifiers import compute_issue_capture, compute_required_coverage
from dronecaptureops.simulation.world import mark_return_status


class ReportTools:
    """Final report and evidence-pack tools."""

    def submit_evidence_pack(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        try:
            report = EvidenceReport(**args)
        except ValidationError as exc:
            return {"accepted": False, "warnings": [str(exc)]}
        world.final_report = report
        world.checklist_status.evidence_submitted = True
        mark_return_status(world)
        score, warnings = validate_evidence_report(world, report)
        cited_coverage, _ = compute_required_coverage(world, cited_only=True)
        cited_issue, _ = compute_issue_capture(world, cited_only=True)
        rows_done = cited_coverage >= 1.0
        anomaly_pairing_done = cited_issue >= 1.0
        world.checklist_status.complete = bool(
            rows_done
            and anomaly_pairing_done
            and world.checklist_status.returned_home
            and world.checklist_status.landed
            and score >= 0.6
        )
        world.done = True
        world.termination_reason = "evidence_submitted"
        return {"accepted": world.checklist_status.complete, "grounding_score": score, "warnings": warnings}
