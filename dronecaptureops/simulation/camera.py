"""Geometry-first camera and capture-quality simulation."""

from __future__ import annotations

from dronecaptureops.core.constants import RGB_EFFECTIVE_RANGE_M, THERMAL_EFFECTIVE_RANGE_M
from dronecaptureops.core.models import Capture, HiddenDefect, SensorType, TargetSurface, Telemetry, WeatherState
from dronecaptureops.utils.math_utils import angle_delta_deg, bearing_deg, clamp, distance_2d


def estimate_visible_targets(
    telemetry: Telemetry,
    targets: list[TargetSurface],
    sensor: SensorType,
    weather: WeatherState,
    hidden_defects: list[HiddenDefect],
    photo_id: str = "ESTIMATE",
    label: str | None = None,
) -> Capture:
    """Return structured capture metadata for the current camera pose."""

    pose = telemetry.pose
    effective_range = THERMAL_EFFECTIVE_RANGE_M if sensor == "thermal" else RGB_EFFECTIVE_RANGE_M
    fov_deg = 72.0 if sensor == "thermal" else 54.0
    camera_yaw = pose.yaw_deg + telemetry.gimbal.yaw_deg

    visible: list[str] = []
    warnings: list[str] = []
    resolution_scores: list[float] = []
    angle_scores: list[float] = []

    for target in targets:
        distance_m = distance_2d(pose, target.center_x, target.center_y)
        yaw_to_target = bearing_deg(pose, target.center_x, target.center_y)
        yaw_delta = angle_delta_deg(camera_yaw, yaw_to_target)
        if distance_m <= effective_range and yaw_delta <= fov_deg / 2.0:
            view_angle = 1.0 - clamp(yaw_delta / (fov_deg / 2.0), 0.0, 1.0)
            resolution = 1.0 - clamp(distance_m / effective_range, 0.0, 1.0) * 0.55
            altitude_penalty = clamp((pose.z - 25.0) / 45.0, 0.0, 0.35)
            visible.append(target.target_id)
            angle_scores.append(clamp(0.35 + 0.65 * view_angle, 0.0, 1.0))
            resolution_scores.append(clamp(resolution - altitude_penalty, 0.0, 1.0))
            if yaw_delta > fov_deg * 0.38:
                warnings.append(f"{target.target_id} near frame edge")

    coverage_pct = 0.0 if not targets else len(visible) / len(targets)
    blur_score = clamp(1.0 - max(0.0, weather.wind_mps - 3.0) * 0.07, 0.45, 1.0)
    occlusion_pct = clamp(max(0.0, 1.0 - weather.visibility) + max(0, len(targets) - len(visible)) * 0.02, 0.0, 0.65)
    resolution_score = sum(resolution_scores) / len(resolution_scores) if resolution_scores else 0.0
    view_angle_score = sum(angle_scores) / len(angle_scores) if angle_scores else 0.0

    detected: list[str] = []
    if sensor == "thermal":
        visible_set = set(visible)
        for defect in hidden_defects:
            if defect.target_id in visible_set and resolution_score >= 0.55 and view_angle_score >= 0.50 and blur_score >= 0.55:
                detected.append(defect.defect_id)

    if not visible:
        warnings.append("no mission targets visible")

    return Capture(
        photo_id=photo_id,
        sensor=sensor,
        label=label,
        pose=pose.model_copy(deep=True),
        gimbal=telemetry.gimbal.model_copy(deep=True),
        targets_visible=visible,
        coverage_pct=round(coverage_pct, 3),
        occlusion_pct=round(occlusion_pct, 3),
        resolution_score=round(resolution_score, 3),
        view_angle_score=round(view_angle_score, 3),
        blur_score=round(blur_score, 3),
        detected_anomalies=detected,
        warnings=warnings,
    )
