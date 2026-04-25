"""Compact text rendering of DroneObservation for LLM agents.

The full Pydantic dump of DroneObservation is ~2-4K tokens — too expensive
to send every step. This module produces a stable, compact text view that
fits in ~500-800 tokens while preserving the fields the agent needs to
plan the next action.
"""

from __future__ import annotations

from typing import Any

from dronecaptureops.core.models import (
    Capture,
    ChecklistStatus,
    DroneObservation,
    InspectionAffordances,
    Telemetry,
)


def render_observation(
    observation: DroneObservation,
    *,
    include_mission: bool = False,
    include_site_map: bool = False,
) -> str:
    """Compact text view of one observation.

    The defaults emit just the per-step state. Pass `include_mission=True`
    on the first turn so the agent sees the mission instruction and
    site geometry once; subsequent turns can omit them since they don't
    change.
    """

    sections: list[str] = []
    sections.append(_render_header(observation))
    if include_mission:
        sections.append(_render_mission(observation))
    if include_site_map:
        sections.append(_render_site_map(observation))
    sections.append(_render_telemetry(observation.telemetry))
    sections.append(_render_checklist(observation.checklist_status, observation.mission))
    sections.append(_render_affordances(observation.inspection_affordances))
    if observation.last_capture is not None:
        sections.append(_render_capture(observation.last_capture))
    if observation.action_result:
        sections.append(_render_action_result(observation))
    if observation.warnings:
        sections.append(_render_warnings(observation.warnings))
    return "\n\n".join(section for section in sections if section).strip()


def render_initial_observation(observation: DroneObservation) -> str:
    """Convenience: render with mission + site map (used on the first turn)."""

    return render_observation(observation, include_mission=True, include_site_map=True)


# --- section renderers --------------------------------------------------------


def _render_header(observation: DroneObservation) -> str:
    state = observation.state_summary
    metadata = observation.metadata or {}
    step = metadata.get("step_count", state.get("remaining_steps"))
    remaining = state.get("remaining_steps")
    phase = observation.inspection_affordances.mission_phase
    parts = [f"# step {step}", f"phase: {phase}"]
    if remaining is not None:
        parts.append(f"steps_remaining: {remaining}")
    if observation.error:
        parts.append(f"last_error: {observation.error}")
    return " | ".join(parts)


def _render_mission(observation: DroneObservation) -> str:
    mission = observation.mission
    if mission is None:
        return ""
    lines: list[str] = ["## Mission"]
    if mission.task_id:
        lines.append(f"task: {mission.task_id} ({mission.task_name})")
    lines.append(f"instruction: {mission.instruction.strip()}")
    lines.append(f"required_rows: {', '.join(mission.required_rows)}")
    lines.append(
        "thresholds: "
        f"min_capture_quality={mission.min_capture_quality:.2f}, "
        f"min_rgb_quality={mission.min_rgb_quality:.2f}, "
        f"min_report_grounding={mission.min_report_grounding_score:.2f}, "
        f"min_battery_at_done={mission.min_battery_at_done_pct:.0f}%"
    )
    if mission.success_criteria:
        lines.append("success_criteria: " + "; ".join(mission.success_criteria))
    if mission.public_constraints:
        lines.append("public_constraints: " + "; ".join(mission.public_constraints))
    return "\n".join(lines)


def _render_site_map(observation: DroneObservation) -> str:
    site_map = observation.site_map
    if site_map is None:
        return ""
    lines: list[str] = ["## Site map"]
    home = site_map.home
    lines.append(f"home: ({home.x:.1f}, {home.y:.1f}, {home.z:.1f})")
    if site_map.assets:
        lines.append("assets:")
        for asset in site_map.assets:
            geom = asset.geometry
            modalities = "/".join(asset.required_modalities) or "-"
            lines.append(
                f"  - {asset.asset_id} {asset.asset_type} centre=({geom.center_x:.0f},{geom.center_y:.0f},{geom.center_z:.0f}) "
                f"normal_yaw={geom.normal_yaw_deg:.0f}° req={modalities}"
            )
    if site_map.airspace_zones:
        lines.append("airspace_zones:")
        for zone in site_map.airspace_zones:
            lines.append(
                f"  - {zone.zone_id} ({zone.zone_type}/{zone.constraint_level}) "
                f"x=[{zone.min_x:.0f},{zone.max_x:.0f}] y=[{zone.min_y:.0f},{zone.max_y:.0f}] "
                f"z<={zone.max_altitude_m:.0f}m — {zone.reason}"
            )
    if site_map.viewpoints:
        lines.append("named_viewpoints:")
        for vp in site_map.viewpoints:
            assets = ", ".join(vp.asset_ids) if vp.asset_ids else "-"
            lines.append(
                f"  - {vp.viewpoint_id} pose=({vp.pose.x:.0f},{vp.pose.y:.0f},{vp.pose.z:.0f},{vp.pose.yaw_deg:.0f}°) "
                f"bucket={vp.standoff_bucket} for=[{assets}]"
            )
    return "\n".join(lines)


def _render_telemetry(telemetry: Telemetry | None) -> str:
    if telemetry is None:
        return ""
    pose = telemetry.pose
    autopilot = telemetry.autopilot
    battery = telemetry.battery
    gimbal = telemetry.gimbal
    camera = telemetry.camera
    lines: list[str] = ["## Telemetry"]
    lines.append(
        f"pose=({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f}, yaw={pose.yaw_deg:.0f}°) "
        f"mode={autopilot.mode} armed={autopilot.armed} in_air={telemetry.in_air} landed={telemetry.landed}"
    )
    lines.append(
        f"battery: {battery.level_pct:.1f}% (voltage {battery.voltage_v:.1f}V) "
        f"weather: {telemetry.weather_band} wind"
    )
    lines.append(
        f"gimbal: pitch={gimbal.pitch_deg:.0f}° yaw={gimbal.yaw_deg:.0f}° "
        f"frame={gimbal.frame_mode} target={gimbal.target_asset_id or '-'}"
    )
    lines.append(
        f"camera: source={camera.active_source} zoom={camera.zoom_level:.1f}× storage_left={camera.storage_remaining}"
    )
    lines.append(
        f"distance_flown={telemetry.distance_flown_m:.1f}m elapsed={telemetry.elapsed_time_s:.1f}s"
    )
    return "\n".join(lines)


def _render_checklist(checklist: ChecklistStatus, mission) -> str:
    required = list(mission.required_rows) if mission is not None else []
    covered = list(checklist.thermal_rows_covered)
    missing = [row for row in required if row not in covered]
    detected = list(checklist.anomalies_detected)
    paired = checklist.anomaly_rgb_pairs
    targets = checklist.anomaly_targets
    lines: list[str] = ["## Checklist"]
    lines.append(
        f"thermal_rows_covered={covered or '[]'} missing={missing or '[]'}"
    )
    if detected:
        anomaly_lines = []
        for anomaly in detected:
            target = targets.get(anomaly, "?")
            rgb_status = paired.get(anomaly)
            paired_str = f"paired={rgb_status}" if rgb_status else "RGB pending"
            anomaly_lines.append(f"{anomaly}@{target} ({paired_str})")
        lines.append("anomalies: " + "; ".join(anomaly_lines))
    else:
        lines.append("anomalies: none flagged yet")
    lines.append(
        f"returned_home={checklist.returned_home} landed={checklist.landed} "
        f"submitted={checklist.evidence_submitted} complete={checklist.complete}"
    )
    if checklist.targets_acknowledged:
        lines.append("targets_acknowledged=" + ", ".join(checklist.targets_acknowledged))
    return "\n".join(lines)


def _render_affordances(affordances: InspectionAffordances) -> str:
    lines: list[str] = ["## Affordances"]
    lines.append(f"phase={affordances.mission_phase}")
    if affordances.blockers:
        lines.append("blockers=" + ", ".join(affordances.blockers))
    if affordances.recommended_action_categories:
        lines.append("recommended_categories=" + ", ".join(affordances.recommended_action_categories))
    if affordances.suggested_tools:
        lines.append("suggested_tools=" + ", ".join(affordances.suggested_tools))
    available = [name for name, ok in affordances.action_availability.items() if ok]
    unavailable = [name for name, ok in affordances.action_availability.items() if not ok]
    if available:
        lines.append("available_now=" + ", ".join(sorted(available)))
    if unavailable:
        lines.append("blocked_now=" + ", ".join(sorted(unavailable)))
    return "\n".join(lines)


def _render_capture(capture: Capture) -> str:
    lines: list[str] = ["## Last capture"]
    lines.append(
        f"{capture.photo_id} sensor={capture.sensor} label={capture.label or '-'} "
        f"quality_score={capture.quality_score:.3f} coverage={capture.coverage_pct:.2f} "
        f"occlusion={capture.occlusion_pct:.2f}"
    )
    if capture.targets_visible:
        per_target = ", ".join(
            f"{target}:{capture.target_quality(target):.2f}" for target in capture.targets_visible
        )
        lines.append(f"targets_visible: {per_target}")
    if capture.detected_anomalies:
        lines.append("detected_anomalies=" + ", ".join(capture.detected_anomalies))
    if capture.warnings:
        lines.append("capture_warnings=" + "; ".join(capture.warnings[:5]))
    return "\n".join(lines)


def _render_action_result(observation: DroneObservation) -> str:
    """Render the small action-result dict the env returns each step.

    Skips heavyweight fields that already appear in other sections (full
    capture metadata) to avoid duplicating tokens.
    """

    skip_keys = {
        "per_target_quality",
        "per_target_metrics",
        "quality_inputs",
        "asset_ids",
        "targets_visible",
        "detected_anomalies",
        "warnings",
        "gimbal",
        "camera",
        "pose",
    }
    result = {key: value for key, value in observation.action_result.items() if key not in skip_keys}
    if not result:
        return ""
    lines: list[str] = ["## Last action result"]
    for key, value in result.items():
        lines.append(f"{key}={_compact_value(value)}")
    return "\n".join(lines)


def _render_warnings(warnings: list[str]) -> str:
    return "## Warnings\n- " + "\n- ".join(warnings[-5:])


def _compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, list):
        if len(value) > 6:
            return f"[{len(value)} items: {value[:3]}...{value[-2:]}]"
        return repr(value)
    if isinstance(value, dict):
        if len(value) > 6:
            return f"{{{len(value)} entries}}"
        return repr(value)
    return repr(value)
