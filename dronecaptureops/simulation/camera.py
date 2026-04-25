"""Geometry-first camera and capture-quality simulation.

The simulator computes per-target quality scores from a single camera pose so
that one capture rarely hits all required rows simultaneously. Coverage,
defect detection, and reward verification all consume these per-target
scores instead of an aggregate quality. This is what forces the agent to
plan multiple viewpoints rather than rely on a single overview shot.
"""

from __future__ import annotations

import math

from dronecaptureops.core.constants import (
    RGB_EFFECTIVE_RANGE_M,
    RGB_HFOV_DEG,
    RGB_VFOV_DEG,
    THERMAL_EFFECTIVE_RANGE_M,
    THERMAL_HFOV_DEG,
    THERMAL_VFOV_DEG,
)
from dronecaptureops.core.models import Capture, HiddenDefect, InspectableAsset, SensorType, Telemetry, WeatherState
from dronecaptureops.utils.math_utils import angle_delta_deg, bearing_deg, clamp, distance_2d


MIN_TARGET_VISIBILITY = 0.15  # below this per-target quality the target is dropped from frame
DEFECT_DETECTION_QUALITY = 0.55
DEFECT_DETECTION_RESOLUTION = 0.55


def estimate_visible_targets(
    telemetry: Telemetry,
    targets: list[InspectableAsset],
    sensor: SensorType,
    weather: WeatherState,
    hidden_defects: list[HiddenDefect],
    photo_id: str = "ESTIMATE",
    label: str | None = None,
) -> Capture:
    """Return structured capture metadata with per-target quality scores."""

    pose = telemetry.pose
    if sensor == "thermal":
        effective_range = THERMAL_EFFECTIVE_RANGE_M
        h_fov_deg = THERMAL_HFOV_DEG
        v_fov_deg = THERMAL_VFOV_DEG
    else:
        effective_range = RGB_EFFECTIVE_RANGE_M
        h_fov_deg = RGB_HFOV_DEG
        v_fov_deg = RGB_VFOV_DEG

    camera_yaw = pose.yaw_deg + telemetry.gimbal.yaw_deg
    camera_pitch = telemetry.gimbal.pitch_deg

    # Global per-frame factors (apply equally to every target visible).
    blur_score = clamp(1.0 - max(0.0, weather.wind_mps - 3.0) * 0.07, 0.45, 1.0)
    weather_occlusion = max(0.0, 1.0 - weather.visibility)
    glare_score = clamp(1.0 - max(0.0, abs(camera_pitch) - 70.0) * 0.01, 0.6, 1.0)
    if camera_pitch > -25.0:
        # Near-horizontal gimbal pointed at solar surfaces is glare-prone.
        glare_score = min(glare_score, clamp(0.55 + abs(camera_pitch) / 90.0, 0.4, 1.0))
    thermal_contrast_score = clamp(
        0.45 + 0.55 * weather.visibility - max(0.0, weather.wind_mps - 5.0) * 0.03,
        0.0,
        1.0,
    )

    visible: list[str] = []
    per_target_quality: dict[str, float] = {}
    per_target_metrics: dict[str, dict[str, float]] = {}
    resolution_scores: list[float] = []
    angle_scores: list[float] = []
    standoff_scores: list[float] = []
    warnings: list[str] = []

    zoom_factor = max(1.0, telemetry.camera.zoom_level)
    range_with_zoom = effective_range * min(zoom_factor, 2.0)

    for target in targets:
        h_distance = distance_2d(pose, target.center_x, target.center_y)
        d3 = math.sqrt(h_distance**2 + (pose.z - target.center_z) ** 2)
        if d3 > range_with_zoom * 1.05:
            continue

        yaw_to_target = bearing_deg(pose, target.center_x, target.center_y)
        yaw_delta = angle_delta_deg(camera_yaw, yaw_to_target)

        if h_distance < 0.05:
            target_pitch = -90.0
        else:
            target_pitch = math.degrees(math.atan2(target.center_z - pose.z, h_distance))
        pitch_delta = abs(target_pitch - camera_pitch)

        # Soft frustum: full credit inside half-FOV, linear falloff to half-FOV
        # again past the edge, zero beyond. Targets fully outside frame don't appear.
        h_factor = clamp(1.0 - yaw_delta / max(h_fov_deg / 2.0, 0.1), 0.0, 1.0)
        v_factor = clamp(1.0 - pitch_delta / max(v_fov_deg / 2.0, 0.1), 0.0, 1.0)
        framing_factor = h_factor * v_factor
        if framing_factor <= 0.0:
            continue

        resolution_quality = clamp(1.0 - d3 / max(range_with_zoom, 1.0), 0.0, 1.0)
        view_angle_quality = clamp(0.35 + 0.65 * h_factor, 0.0, 1.0)
        pitch_quality = clamp(0.35 + 0.65 * v_factor, 0.0, 1.0)
        standoff_quality = _score_standoff(target, d3)
        altitude_penalty = clamp((pose.z - 30.0) / 50.0, 0.0, 0.30)
        scene_occlusion = clamp(weather_occlusion + 0.05 * (1.0 - framing_factor), 0.0, 0.6)

        per_target_q = clamp(
            0.30 * resolution_quality
            + 0.20 * view_angle_quality
            + 0.15 * pitch_quality
            + 0.15 * standoff_quality
            + 0.10 * blur_score
            + 0.10 * (1.0 - scene_occlusion)
            - altitude_penalty,
            0.0,
            1.0,
        )

        if per_target_q < MIN_TARGET_VISIBILITY:
            continue

        visible.append(target.asset_id)
        per_target_quality[target.asset_id] = round(per_target_q, 4)
        per_target_metrics[target.asset_id] = {
            "distance_m": round(d3, 2),
            "yaw_delta_deg": round(yaw_delta, 2),
            "pitch_delta_deg": round(pitch_delta, 2),
            "framing_factor": round(framing_factor, 3),
            "resolution_quality": round(resolution_quality, 3),
            "view_angle_quality": round(view_angle_quality, 3),
            "pitch_quality": round(pitch_quality, 3),
            "standoff_quality": round(standoff_quality, 3),
        }
        resolution_scores.append(resolution_quality)
        angle_scores.append(view_angle_quality)
        standoff_scores.append(standoff_quality)
        if framing_factor < 0.45:
            warnings.append(f"{target.asset_id} near frame edge")

    coverage_pct = 0.0 if not targets else len(visible) / len(targets)
    occlusion_pct = clamp(
        weather_occlusion + max(0, len(targets) - len(visible)) * 0.02,
        0.0,
        0.65,
    )
    resolution_score = sum(resolution_scores) / len(resolution_scores) if resolution_scores else 0.0
    view_angle_score = sum(angle_scores) / len(angle_scores) if angle_scores else 0.0
    standoff_score = sum(standoff_scores) / len(standoff_scores) if standoff_scores else 0.0
    gsd_score = clamp(resolution_score * (0.85 + 0.05 * telemetry.camera.zoom_level), 0.0, 1.0)

    detected = _detect_anomalies(
        sensor=sensor,
        per_target_quality=per_target_quality,
        per_target_metrics=per_target_metrics,
        hidden_defects=hidden_defects,
        glare_score=glare_score,
        blur_score=blur_score,
        thermal_contrast_score=thermal_contrast_score,
    )

    if not visible:
        warnings.append("no mission targets visible")

    return Capture(
        photo_id=photo_id,
        sensor=sensor,
        label=label,
        pose=pose.model_copy(deep=True),
        gimbal=telemetry.gimbal.model_copy(deep=True),
        camera=telemetry.camera.model_copy(deep=True),
        asset_ids=visible,
        targets_visible=visible,
        coverage_pct=round(coverage_pct, 3),
        occlusion_pct=round(occlusion_pct, 3),
        resolution_score=round(resolution_score, 3),
        view_angle_score=round(view_angle_score, 3),
        blur_score=round(blur_score, 3),
        standoff_score=round(standoff_score, 3),
        gsd_score=round(gsd_score, 3),
        glare_score=round(glare_score, 3),
        thermal_contrast_score=round(thermal_contrast_score, 3),
        detected_anomalies=detected,
        per_target_quality=per_target_quality,
        per_target_metrics=per_target_metrics,
        quality_inputs={
            "standoff_score": round(standoff_score, 3),
            "gsd_score": round(gsd_score, 3),
            "glare_score": round(glare_score, 3),
            "thermal_contrast_score": round(thermal_contrast_score, 3),
        },
        warnings=warnings,
    )


def _detect_anomalies(
    *,
    sensor: SensorType,
    per_target_quality: dict[str, float],
    per_target_metrics: dict[str, dict[str, float]],
    hidden_defects: list[HiddenDefect],
    glare_score: float,
    blur_score: float,
    thermal_contrast_score: float,
) -> list[str]:
    """Return defect IDs that the simulator surfaces in this capture.

    Detection is per-target: a defect is flagged only when the underlying
    target is captured at sufficient quality, with sensor-specific
    requirements baked into HiddenDefect. False-positive defects are
    surfaced when their conditions are met (e.g. glare artifacts when
    glare_score is low) so the agent must reason about them.
    """

    detected: list[str] = []
    for defect in hidden_defects:
        target_q = per_target_quality.get(defect.target_id, 0.0)
        metrics = per_target_metrics.get(defect.target_id, {})

        if defect.defect_type == "false_thermal_artifact":
            # Glare-induced false positive: only appears when the gimbal pitch
            # creates significant glare AND the sensor is thermal AND the
            # target frames in. Disappears at steeper angles.
            if sensor != "thermal":
                continue
            if target_q < MIN_TARGET_VISIBILITY:
                continue
            if glare_score >= 0.78:
                continue
            detected.append(defect.defect_id)
            continue

        if defect.defect_type == "vegetation_shadow":
            # Shadows show in thermal at oblique angles; an overhead capture
            # washes them out. Require non-overhead pitch alignment.
            if sensor != "thermal":
                continue
            if target_q < DEFECT_DETECTION_QUALITY:
                continue
            pitch_delta = metrics.get("pitch_delta_deg", 0.0)
            if pitch_delta >= 6.0 and target_q >= defect.min_quality:
                detected.append(defect.defect_id)
            continue

        if defect.defect_type == "bypass_diode_fault":
            # Diode faults need close-standoff thermal evidence and decent
            # contrast (good lighting, low wind).
            if sensor != "thermal":
                continue
            if target_q < defect.min_quality:
                continue
            distance_m = metrics.get("distance_m", 9999.0)
            if distance_m > 32.0:
                continue
            if thermal_contrast_score < 0.6:
                continue
            detected.append(defect.defect_id)
            continue

        if defect.defect_type == "soiling_heating":
            # Soiling is detectable on thermal at any reasonable standoff
            # provided the target is well-framed and not too windy.
            if sensor != "thermal":
                continue
            if target_q < defect.min_quality:
                continue
            if blur_score < 0.6:
                continue
            detected.append(defect.defect_id)
            continue

        # Default: thermal hotspot — visible whenever the target is captured
        # at the configured min_quality on the required sensor.
        if sensor != defect.required_sensor:
            continue
        if target_q < defect.min_quality:
            continue
        resolution_quality = metrics.get("resolution_quality", 0.0)
        if resolution_quality < DEFECT_DETECTION_RESOLUTION:
            continue
        detected.append(defect.defect_id)

    return detected


def _score_standoff(target: InspectableAsset, distance_m: float) -> float:
    if not target.safe_standoff_bands:
        return 1.0
    best = 0.0
    for band in target.safe_standoff_bands:
        if band.min_m <= distance_m <= band.max_m:
            error = abs(distance_m - band.preferred_m) / max(band.max_m - band.min_m, 1.0)
            best = max(best, clamp(1.0 - error, 0.0, 1.0))
        else:
            # Soft penalty when outside the band: still allow some signal.
            outside = min(abs(distance_m - band.min_m), abs(distance_m - band.max_m))
            best = max(best, clamp(0.5 - outside / 30.0, 0.0, 0.45))
    return best
