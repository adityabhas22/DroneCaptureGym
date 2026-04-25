"""Camera tool handlers."""

from __future__ import annotations

import math
from typing import Any

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.core.coercion import coerce_float, coerce_optional_float, coerce_str
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import Pose3D, SensorType
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.simulation.camera import estimate_visible_targets
from dronecaptureops.simulation.safety import SafetyChecker
from dronecaptureops.utils.math_utils import bearing_deg, distance_2d


_CAMERA_SOURCES: set[str] = {"rgb", "thermal", "rgb_thermal"}
_SENSOR_OPTIONS: set[str] = {"rgb", "thermal"}


class CameraTools:
    """High-level camera and gimbal tools."""

    def __init__(self, controller: DroneController, safety: SafetyChecker) -> None:
        self._controller = controller
        self._safety = safety

    def set_gimbal(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        pitch = coerce_float(args, "pitch_deg")
        yaw = coerce_optional_float(args, "yaw_deg")
        self._safety.validate_gimbal(pitch, yaw)
        return self._controller.set_gimbal(world, pitch, yaw)

    def set_zoom(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        zoom_level = coerce_float(args, "zoom_level", minimum=1.0, maximum=4.0)
        world.telemetry.camera.zoom_level = zoom_level
        world.telemetry.autopilot.last_command = "set_zoom"
        world.telemetry.autopilot.command_status = "completed"
        world.telemetry.sync_legacy_fields()
        return world.telemetry.camera.model_dump(mode="json")

    def set_camera_source(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        source = coerce_str(args, "source", allowed=_CAMERA_SOURCES)
        if source not in world.telemetry.camera.supported_sources:
            raise ActionValidationError(f"unsupported camera source: {source}")
        world.telemetry.camera.active_source = source  # type: ignore[assignment]
        world.telemetry.autopilot.last_command = "set_camera_source"
        world.telemetry.autopilot.command_status = "completed"
        world.telemetry.sync_legacy_fields()
        return world.telemetry.camera.model_dump(mode="json")

    def point_camera_at(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        asset_id = coerce_str(args, "asset_id")
        asset = next((item for item in world.assets if item.asset_id == asset_id), None)
        if asset is None:
            raise ActionValidationError(f"unknown asset_id: {asset_id}")
        bearing = bearing_deg(world.telemetry.pose, asset.center_x, asset.center_y)
        body_yaw = (bearing - world.telemetry.pose.yaw_deg + 180.0) % 360.0 - 180.0
        distance_m = max(distance_2d(world.telemetry.pose, asset.center_x, asset.center_y), 0.1)
        pitch = -min(85.0, max(25.0, math.degrees(math.atan2(world.telemetry.pose.z, distance_m))))
        self._safety.validate_gimbal(pitch, body_yaw)
        self._controller.set_gimbal(world, pitch, body_yaw)
        world.telemetry.gimbal.frame_mode = "roi"
        world.telemetry.gimbal.target_asset_id = asset_id
        world.telemetry.gimbal.roi = Pose3D(
            x=asset.center_x,
            y=asset.center_y,
            z=asset.center_z,
            yaw_deg=asset.normal_yaw_deg,
        )
        return world.telemetry.gimbal.model_dump(mode="json")

    def capture_rgb(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        return self._capture(world, "rgb", args)

    def capture_thermal(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        return self._capture(world, "thermal", args)

    def inspect_capture(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        photo_id = coerce_str(args, "photo_id")
        for capture in world.capture_log:
            if capture.photo_id == photo_id:
                if photo_id not in world.inspected_photo_ids:
                    world.inspected_photo_ids.append(photo_id)
                return capture.model_dump(mode="json") | {"quality_score": capture.quality_score}
        return {"error": f"unknown photo_id {photo_id}"}

    def estimate_view(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        sensor_str = coerce_str(args, "sensor", default="thermal", allowed=_SENSOR_OPTIONS)
        sensor: SensorType = "thermal" if sensor_str == "thermal" else "rgb"
        estimate = estimate_visible_targets(
            telemetry=world.telemetry,
            targets=world.assets,
            sensor=sensor,
            weather=world.weather,
            hidden_defects=[],
        )
        return estimate.model_dump(mode="json") | {"quality_score": estimate.quality_score}

    def _capture(self, world: EpisodeWorld, sensor: SensorType, args: dict[str, Any]) -> dict[str, Any]:
        from dronecaptureops.core.errors import SafetyViolationError
        from dronecaptureops.simulation.world import active_zones

        label_raw = args.get("label")
        if label_raw is not None and not isinstance(label_raw, str):
            raise ActionValidationError(f"invalid label: expected string, got {type(label_raw).__name__}")

        # Privacy zones block image capture from inside, even though they
        # don't block flight. This is the active-inspection equivalent of
        # respecting a "no photo" perimeter on a real site.
        pose = world.telemetry.pose
        for zone in active_zones(world):
            if zone.zone_type != "privacy":
                continue
            if not (zone.min_x <= pose.x <= zone.max_x and zone.min_y <= pose.y <= zone.max_y):
                continue
            if not (zone.min_altitude_m <= pose.z <= zone.max_altitude_m):
                continue
            raise SafetyViolationError(f"privacy_capture_violation:{zone.zone_id}")

        world.telemetry.camera.active_source = sensor
        capture = self._controller.capture_image(world, sensor, label_raw)
        self._update_checklist_from_capture(world, capture)
        return capture.model_dump(mode="json") | {"quality_score": capture.quality_score}

    def _update_checklist_from_capture(self, world: EpisodeWorld, capture) -> None:
        thermal_threshold = world.mission.min_capture_quality
        rgb_threshold = world.mission.min_rgb_quality
        if capture.sensor == "thermal":
            covered = set(world.checklist_status.thermal_rows_covered)
            for target_id in capture.targets_visible:
                if target_id in world.mission.required_rows and capture.target_quality(target_id) >= thermal_threshold:
                    covered.add(target_id)
            world.checklist_status.thermal_rows_covered = sorted(covered)
            anomalies = set(world.checklist_status.anomalies_detected)
            anomalies.update(capture.detected_anomalies)
            world.checklist_status.anomalies_detected = sorted(anomalies)
            # Surface (anomaly_id → target_id) mapping only after the simulator
            # actually flagged the defect; this is the agent's first signal that
            # the defect exists.
            for anomaly in capture.detected_anomalies:
                defect = next((item for item in world.hidden_defects if item.defect_id == anomaly), None)
                if defect is not None:
                    world.checklist_status.anomaly_targets.setdefault(anomaly, defect.target_id)
        if capture.sensor == "rgb":
            for anomaly in world.checklist_status.anomalies_detected:
                defect = next((item for item in world.hidden_defects if item.defect_id == anomaly), None)
                if defect is None or defect.target_id not in capture.targets_visible:
                    continue
                if capture.target_quality(defect.target_id) >= rgb_threshold:
                    world.checklist_status.anomaly_rgb_pairs.setdefault(anomaly, capture.photo_id)
