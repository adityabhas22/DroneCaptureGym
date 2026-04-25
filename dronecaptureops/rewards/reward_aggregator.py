"""Composable reward aggregation."""

from __future__ import annotations

from dataclasses import dataclass

from dronecaptureops.core.models import RawDroneAction, RewardBreakdown
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.capture_quality import CaptureQualityReward
from dronecaptureops.rewards.efficiency import BatteryManagementReward
from dronecaptureops.rewards.report_grounding import ReportGroundingReward
from dronecaptureops.rewards.safety import SafetyReward
from dronecaptureops.rewards.verifiers import (
    compute_evidence_success,
    compute_integrity_gate,
    compute_issue_capture,
    compute_operational_efficiency,
    compute_photo_value,
    compute_required_coverage,
    compute_safety_gate,
    compute_value_per_photo,
)
from dronecaptureops.utils.math_utils import clamp


@dataclass(frozen=True)
class RewardStepContext:
    """Optional step transition context used for small process rewards."""

    previous_world: EpisodeWorld
    action: RawDroneAction
    success: bool
    format_valid: bool
    result: dict


class RewardAggregator:
    """Computes normalized reward components and total reward."""

    def __init__(self) -> None:
        self._quality = CaptureQualityReward()
        self._safety = SafetyReward()
        self._battery = BatteryManagementReward()
        self._report = ReportGroundingReward()

    def compute(
        self,
        world: EpisodeWorld,
        format_valid: bool = True,
        context: RewardStepContext | None = None,
    ) -> RewardBreakdown:
        """Compute and store the current reward breakdown."""

        terminal_submitted = world.final_report is not None
        captured_coverage, coverage_debug = compute_required_coverage(world)
        captured_issue, issue_debug = compute_issue_capture(world)
        cited_coverage, cited_coverage_debug = compute_required_coverage(world, cited_only=True)
        cited_issue, cited_issue_debug = compute_issue_capture(world, cited_only=True)
        required_coverage = cited_coverage if terminal_submitted else captured_coverage
        issue_capture = cited_issue if terminal_submitted else captured_issue
        evidence_success = compute_evidence_success(world, required_coverage, issue_capture)
        captured_evidence_success = compute_evidence_success(world, captured_coverage, captured_issue)
        operational_efficiency = compute_operational_efficiency(world, evidence_success)
        grounded_report = self._report.compute(world)
        capture_quality = self._quality.compute(world)
        safety_compliance = self._safety.compute(world)
        battery_management = self._battery.compute(world)
        checklist_completion = self._checklist_completion(world)
        recovery_behavior = self._recovery_behavior(world)
        value_per_photo = compute_value_per_photo(world)
        safety_gate = compute_safety_gate(world)
        integrity_gate, integrity_warnings = compute_integrity_gate(world)
        process_reward = self._process_reward(world, context)
        penalties = self._penalties(world, format_valid)

        raw_outcome = (
            0.45 * evidence_success
            + 0.20 * required_coverage
            + 0.15 * issue_capture
            + 0.10 * operational_efficiency
            + 0.10 * grounded_report
            + process_reward
            - penalties
        )
        raw_outcome_if_submitted = (
            0.45 * captured_evidence_success
            + 0.20 * captured_coverage
            + 0.15 * captured_issue
            + 0.10 * compute_operational_efficiency(world, captured_evidence_success)
            + 0.10 * grounded_report
            + process_reward
            - penalties
        )
        shaping_reward = self._shaping_reward(captured_evidence_success, captured_coverage, captured_issue, process_reward, penalties)
        if terminal_submitted:
            capped_total = min(raw_outcome, safety_gate, integrity_gate)
        else:
            capped_total = shaping_reward
        debug = {
            **coverage_debug,
            **issue_debug,
            "terminal_submitted": terminal_submitted,
            "nonterminal_cap_applied": not terminal_submitted,
            "shaping_reward": round(shaping_reward, 4),
            "raw_outcome_if_submitted": round(raw_outcome_if_submitted, 4),
            "captured_required_coverage": captured_coverage,
            "captured_issue_capture": captured_issue,
            "cited_required_coverage": cited_coverage,
            "cited_issue_capture": cited_issue,
            "missing_cited_rows": cited_coverage_debug["missing_rows"],
            "cited_issue_details": cited_issue_debug["issue_details"],
            "photos_taken": len(world.capture_log),
            "valid_photos": len([capture for capture in world.capture_log if capture.targets_visible and capture.quality_score >= 0.55]),
            "battery_remaining": round(world.telemetry.battery.level_pct, 3),
            "distance_flown_m": round(world.distance_flown_m, 3),
            "elapsed_time_s": round(world.elapsed_time_s, 3),
            "integrity_warnings": integrity_warnings,
            "raw_total_before_caps": round(raw_outcome, 4),
        }
        breakdown = RewardBreakdown(
            format_validity=1.0 if format_valid else 0.0,
            flight_success=1.0 if world.telemetry.in_air or world.telemetry.landed else 0.0,
            evidence_success=evidence_success,
            required_coverage=required_coverage,
            issue_capture=issue_capture,
            operational_efficiency=operational_efficiency,
            grounded_report=grounded_report,
            process_reward=process_reward,
            integrity_gate=integrity_gate,
            value_per_photo=value_per_photo,
            target_coverage=required_coverage,
            capture_quality=capture_quality,
            defect_visibility=issue_capture,
            checklist_completion=checklist_completion,
            route_efficiency=operational_efficiency,
            battery_management=battery_management,
            safety_compliance=safety_compliance,
            report_grounding=grounded_report,
            recovery_behavior=recovery_behavior,
            penalties=round(penalties, 4),
            safety_gate=safety_gate,
            total=round(clamp(capped_total, -1.0, 1.0), 4),
            debug=debug,
        )
        world.reward_breakdown = breakdown
        return breakdown

    def _checklist_completion(self, world: EpisodeWorld) -> float:
        required = set(world.mission.required_rows)
        covered = set(world.checklist_status.thermal_rows_covered)
        parts = [
            1.0 if required and required <= covered else len(required & covered) / max(len(required), 1),
            1.0 if not world.checklist_status.anomalies_detected or world.checklist_status.anomaly_rgb_pairs else 0.0,
            1.0 if world.checklist_status.returned_home else 0.0,
            1.0 if world.checklist_status.landed else 0.0,
            1.0 if world.checklist_status.evidence_submitted else 0.0,
        ]
        return round(sum(parts) / len(parts), 4)

    def _recovery_behavior(self, world: EpisodeWorld) -> float:
        """Reward both flawless work and active recovery from a bad shot.

        Previously a flawless run scored 0.5 and a recovery scored 1.0 — which
        perversely punished agents that got it right the first time. Now
        flawless work scores 1.0; if any low-quality capture exists, full
        credit requires a later high-quality capture (the recovery), and
        partial credit (0.3) is given to acknowledge the attempt.
        """

        if not world.capture_log:
            return 0.0
        low_quality_indices = [
            idx
            for idx, capture in enumerate(world.capture_log)
            if capture.targets_visible and capture.quality_score < 0.55
        ]
        if not low_quality_indices:
            return 1.0
        last_low = low_quality_indices[-1]
        later_good = any(capture.quality_score >= 0.68 for capture in world.capture_log[last_low + 1:])
        return 1.0 if later_good else 0.3

    def _process_reward(self, world: EpisodeWorld, context: RewardStepContext | None) -> float:
        if context is None:
            return round(world.process_reward_total, 4)
        bonus = 0.0
        action_name = context.action.tool_name
        previous = context.previous_world
        if context.success and action_name in {"capture_thermal", "capture_rgb"} and len(world.capture_log) > len(previous.capture_log):
            capture = world.capture_log[-1]
            previous_coverage, _ = compute_required_coverage(previous)
            current_coverage, _ = compute_required_coverage(world)
            previous_issue, _ = compute_issue_capture(previous)
            current_issue, _ = compute_issue_capture(world)
            if current_coverage > previous_coverage:
                bonus += 0.020
            if capture.sensor == "thermal" and current_coverage > previous_coverage:
                bonus += 0.015
            if capture.sensor == "rgb" and current_issue > previous_issue:
                bonus += 0.015
            previous_best = max(
                [old.quality_score for old in previous.capture_log if set(old.targets_visible) & set(capture.targets_visible)] or [0.0]
            )
            if previous_best < 0.55 <= capture.quality_score:
                bonus += 0.030
        if (
            context.success
            and action_name == "return_home"
            and not previous.checklist_status.returned_home
            and world.checklist_status.returned_home
            and world.telemetry.battery.level_pct >= world.mission.min_battery_at_done_pct
        ):
            bonus += 0.030
        inspected_photo_id = context.action.arguments.get("photo_id")
        if (
            context.success
            and action_name == "inspect_capture"
            and inspected_photo_id in {capture.photo_id for capture in world.capture_log}
            and inspected_photo_id not in previous.inspected_photo_ids
        ):
            bonus += 0.010
        world.process_reward_total = round(min(0.10, world.process_reward_total + bonus), 4)
        return world.process_reward_total

    def _shaping_reward(
        self,
        captured_evidence_success: float,
        captured_coverage: float,
        captured_issue: float,
        process_reward: float,
        penalties: float,
    ) -> float:
        shaping = (
            0.10 * captured_evidence_success
            + 0.04 * captured_coverage
            + 0.04 * captured_issue
            + process_reward
            - penalties
        )
        return round(clamp(shaping, 0.0, 0.20), 4)

    def _penalties(self, world: EpisodeWorld, format_valid: bool) -> float:
        penalties = 0.0
        if not format_valid:
            penalties += 0.05
        penalties += 0.05 * max(world.invalid_action_count, 0)
        low_value_redundant = sum(
            1
            for index, capture in enumerate(world.capture_log)
            if compute_photo_value(world, capture) < 0.05
            and any(
                prior.sensor == capture.sensor and set(prior.targets_visible) == set(capture.targets_visible)
                for prior in world.capture_log[:index]
            )
        )
        penalties += 0.02 * low_value_redundant
        penalties += self._indiscriminate_citation_penalty(world)
        if world.done and world.mission.must_return_home and not world.checklist_status.returned_home:
            penalties += 0.20
        if world.final_report is not None and not world.final_report.photo_ids and not world.final_report.evidence and not world.final_report.issues_found:
            penalties += 0.05
        return penalties

    def _indiscriminate_citation_penalty(self, world: EpisodeWorld) -> float:
        """Penalize 'cite every photo' reports.

        Counts cited captures that bring no new evidence (no new row coverage,
        no new anomaly, no RGB context for a real defect, no quality
        improvement over earlier captures of the same target). Triggers only
        after submission and only when filler dominates. Capped at 0.10 so it
        can't dominate outcome reward.
        """

        from dronecaptureops.rewards.verifiers import report_cited_photo_ids, reportable_defects

        report = world.final_report
        if report is None:
            return 0.0
        cited_ids = report_cited_photo_ids(report)
        cited_captures = [capture for capture in world.capture_log if capture.photo_id in cited_ids]
        if len(cited_captures) <= 3:
            return 0.0
        required = set(world.mission.required_rows)
        reportable_targets = {defect.target_id for defect in reportable_defects(world)}
        reportable_ids = {defect.defect_id for defect in reportable_defects(world)}
        covered: set[str] = set()
        detected: set[str] = set()
        best_quality: dict[tuple[str, str], float] = {}
        filler = 0
        for capture in cited_captures:
            new_rows = {
                row for row in capture.targets_visible
                if row in required and capture.sensor == "thermal"
                and capture.target_quality(row) >= world.mission.min_capture_quality
                and row not in covered
            }
            new_anomalies = {a for a in capture.detected_anomalies if a in reportable_ids and a not in detected}
            rgb_context = (
                capture.sensor == "rgb"
                and bool(set(capture.targets_visible) & reportable_targets)
            )
            quality_jump = False
            for target in capture.targets_visible:
                key = (capture.sensor, target)
                prior_best = best_quality.get(key, 0.0)
                if capture.target_quality(target) > prior_best + 0.05:
                    quality_jump = True
                best_quality[key] = max(prior_best, capture.target_quality(target))
            covered |= new_rows
            detected |= new_anomalies
            if not (new_rows or new_anomalies or rgb_context or quality_jump):
                filler += 1
        if filler == 0:
            return 0.0
        excess = max(0, len(cited_captures) - 3)
        filler_ratio = filler / len(cited_captures)
        if filler_ratio < 0.34:
            return 0.0
        return round(min(0.10, 0.02 * filler * filler_ratio + 0.005 * excess), 4)
