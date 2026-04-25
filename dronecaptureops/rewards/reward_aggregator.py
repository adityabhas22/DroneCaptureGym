"""Composable reward aggregation."""

from __future__ import annotations

from dronecaptureops.core.models import RewardBreakdown
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.rewards.capture_quality import CaptureQualityReward
from dronecaptureops.rewards.coverage import TargetCoverageReward
from dronecaptureops.rewards.efficiency import BatteryManagementReward, RouteEfficiencyReward
from dronecaptureops.rewards.report_grounding import ReportGroundingReward
from dronecaptureops.rewards.safety import SafetyReward
from dronecaptureops.utils.math_utils import clamp


class RewardAggregator:
    """Computes normalized reward components and total reward."""

    def __init__(self) -> None:
        self._coverage = TargetCoverageReward()
        self._quality = CaptureQualityReward()
        self._safety = SafetyReward()
        self._route = RouteEfficiencyReward()
        self._battery = BatteryManagementReward()
        self._report = ReportGroundingReward()

    def compute(self, world: EpisodeWorld, format_valid: bool = True) -> RewardBreakdown:
        """Compute and store the current reward breakdown."""

        target_coverage = self._coverage.compute(world)
        capture_quality = self._quality.compute(world)
        safety_compliance = self._safety.compute(world)
        route_efficiency = self._route.compute(world)
        battery_management = self._battery.compute(world)
        report_grounding = self._report.compute(world)
        defect_visibility = self._defect_visibility(world)
        checklist_completion = self._checklist_completion(world)
        recovery_behavior = self._recovery_behavior(world)
        penalties = self._penalties(world, format_valid)
        safety_gate = 0.0 if any(
            "no_fly" in violation
            or "obstacle" in violation
            or "privacy" in violation
            or "battery_exhausted" in violation
            for violation in world.safety_violations
        ) else 1.0
        total = safety_gate * (
            0.25 * target_coverage
            + 0.20 * capture_quality
            + 0.15 * defect_visibility
            + 0.10 * checklist_completion
            + 0.10 * route_efficiency
            + 0.10 * battery_management
            + 0.05 * report_grounding
            + 0.05 * recovery_behavior
        ) - penalties
        breakdown = RewardBreakdown(
            format_validity=1.0 if format_valid else 0.0,
            flight_success=1.0 if world.telemetry.in_air or world.telemetry.landed else 0.0,
            target_coverage=target_coverage,
            capture_quality=capture_quality,
            defect_visibility=defect_visibility,
            checklist_completion=checklist_completion,
            route_efficiency=route_efficiency,
            battery_management=battery_management,
            safety_compliance=safety_compliance,
            report_grounding=report_grounding,
            recovery_behavior=recovery_behavior,
            penalties=round(penalties, 4),
            safety_gate=safety_gate,
            total=round(clamp(total, -1.0, 1.0), 4),
        )
        world.reward_breakdown = breakdown
        return breakdown

    def _defect_visibility(self, world: EpisodeWorld) -> float:
        hidden_ids = {defect.defect_id for defect in world.hidden_defects if defect.defect_type == "thermal_hotspot"}
        if not hidden_ids:
            return 1.0
        detected = set(world.checklist_status.anomalies_detected)
        return round(len(hidden_ids & detected) / len(hidden_ids), 4)

    def _checklist_completion(self, world: EpisodeWorld) -> float:
        required = set(world.mission.required_rows)
        covered = set(world.checklist_status.thermal_rows_covered)
        detected = set(world.checklist_status.anomalies_detected)
        if not world.mission.rgb_closeup_for_anomalies:
            anomaly_completion = 1.0
        elif not detected:
            anomaly_completion = 1.0
        else:
            anomaly_completion = len(detected & set(world.checklist_status.anomaly_rgb_pairs)) / len(detected)
        parts = [
            1.0 if required and required <= covered else len(required & covered) / max(len(required), 1),
            anomaly_completion,
            1.0 if world.checklist_status.returned_home else 0.0,
            1.0 if world.checklist_status.landed else 0.0,
            1.0 if world.checklist_status.evidence_submitted else 0.0,
        ]
        return round(sum(parts) / len(parts), 4)

    def _recovery_behavior(self, world: EpisodeWorld) -> float:
        if not world.capture_log:
            return 0.0
        low_quality = [capture for capture in world.capture_log if capture.targets_visible and capture.quality_score < 0.55]
        if not low_quality:
            return 0.5
        later_good = any(capture.quality_score >= 0.68 for capture in world.capture_log[len(low_quality):])
        return 1.0 if later_good else 0.0

    def _penalties(self, world: EpisodeWorld, format_valid: bool) -> float:
        penalties = 0.0
        if not format_valid:
            penalties += 0.12
        penalties += 0.10 * world.invalid_action_count
        redundant = max(0, len(world.capture_log) - len({tuple(capture.targets_visible) + (capture.sensor,) for capture in world.capture_log}))
        penalties += 0.03 * redundant
        if world.done and world.mission.must_return_home and not world.checklist_status.returned_home:
            penalties += 0.20
        return penalties
