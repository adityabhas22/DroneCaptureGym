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

        required_coverage, coverage_debug = compute_required_coverage(world)
        issue_capture, issue_debug = compute_issue_capture(world)
        evidence_success = compute_evidence_success(world, required_coverage, issue_capture)
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

        raw_total = (
            0.45 * evidence_success
            + 0.20 * required_coverage
            + 0.15 * issue_capture
            + 0.10 * operational_efficiency
            + 0.10 * grounded_report
            + process_reward
            - penalties
        )
        capped_total = min(raw_total, safety_gate, integrity_gate)
        debug = {
            **coverage_debug,
            **issue_debug,
            "photos_taken": len(world.capture_log),
            "valid_photos": len([capture for capture in world.capture_log if capture.targets_visible and capture.quality_score >= 0.55]),
            "battery_remaining": round(world.telemetry.battery.level_pct, 3),
            "distance_flown_m": round(world.distance_flown_m, 3),
            "elapsed_time_s": round(world.elapsed_time_s, 3),
            "integrity_warnings": integrity_warnings,
            "raw_total_before_caps": round(raw_total, 4),
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
        if not world.capture_log:
            return 0.0
        low_quality = [capture for capture in world.capture_log if capture.targets_visible and capture.quality_score < 0.55]
        if not low_quality:
            return 0.5
        later_good = any(capture.quality_score >= 0.68 for capture in world.capture_log[len(low_quality):])
        return 1.0 if later_good else 0.0

    def _process_reward(self, world: EpisodeWorld, context: RewardStepContext | None) -> float:
        if context is None:
            return round(world.process_reward_total, 4)
        bonus = 0.0
        action_name = context.action.tool_name
        previous = context.previous_world
        if context.format_valid:
            bonus += 0.005
        if context.success and action_name in {"takeoff", "fly_to_viewpoint", "return_home", "land"}:
            bonus += 0.005
        if context.success and action_name in {"capture_thermal", "capture_rgb"} and len(world.capture_log) > len(previous.capture_log):
            capture = world.capture_log[-1]
            if capture.targets_visible and capture.quality_score >= 0.55:
                bonus += 0.020
            if capture.sensor == "thermal" and set(capture.targets_visible) & set(world.mission.required_rows):
                bonus += 0.015
            if capture.sensor == "rgb" and world.checklist_status.anomalies_detected:
                bonus += 0.015
            previous_best = max(
                [old.quality_score for old in previous.capture_log if set(old.targets_visible) & set(capture.targets_visible)] or [0.0]
            )
            if previous_best < 0.55 <= capture.quality_score:
                bonus += 0.030
        if context.success and action_name == "return_home" and world.telemetry.battery.level_pct >= world.mission.min_battery_at_done_pct:
            bonus += 0.030
        if context.success and action_name == "inspect_capture":
            bonus += 0.010
        world.process_reward_total = round(min(0.10, world.process_reward_total + bonus), 4)
        return world.process_reward_total

    def _penalties(self, world: EpisodeWorld, format_valid: bool) -> float:
        penalties = 0.0
        if not format_valid:
            penalties += 0.05
        penalties += 0.05 * max(world.invalid_action_count, 0)
        redundant = max(0, len(world.capture_log) - len({tuple(capture.targets_visible) + (capture.sensor,) for capture in world.capture_log}))
        penalties += 0.02 * redundant
        if world.done and world.mission.must_return_home and not world.checklist_status.returned_home:
            penalties += 0.20
        if world.final_report is not None and not world.final_report.photo_ids and not world.final_report.evidence and not world.final_report.issues_found:
            penalties += 0.05
        return penalties
