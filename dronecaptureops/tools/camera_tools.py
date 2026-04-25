"""Camera tool handlers."""

from __future__ import annotations

import math
from typing import Any

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import Pose3D, SensorType
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.simulation.camera import estimate_visible_targets
from dronecaptureops.simulation.safety import SafetyChecker
from dronecaptureops.utils.math_utils import bearing_deg, distance_2d


class CameraTools:
    """High-level camera and gimbal tools."""

    def __init__(self, controller: DroneController, safety: SafetyChecker) -> None:
        self._controller = controller
        self._safety = safety

    def set_gimbal(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        pitch = float(args["pitch_deg"])
        yaw = args.get("yaw_deg")
        yaw_value = None if yaw is None else float(yaw)
        self._safety.validate_gimbal(pitch, yaw_value)
        return self._controller.set_gimbal(world, pitch, yaw_value)

    def set_zoom(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        zoom_level = max(1.0, min(float(args["zoom_level"]), 4.0))
        world.telemetry.camera.zoom_level = zoom_level
        world.telemetry.autopilot.last_command = "set_zoom"
        world.telemetry.autopilot.command_status = "completed"
        world.telemetry.sync_legacy_fields()
        return world.telemetry.camera.model_dump(mode="json")

    def set_camera_source(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        source = args["source"]
        if source not in world.telemetry.camera.supported_sources:
            raise ActionValidationError(f"unsupported camera source: {source}")
        world.telemetry.camera.active_source = source
        world.telemetry.autopilot.last_command = "set_camera_source"
        world.telemetry.autopilot.command_status = "completed"
        world.telemetry.sync_legacy_fields()
        return world.telemetry.camera.model_dump(mode="json")

    def point_camera_at(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        asset_id = args["asset_id"]
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
        photo_id = args["photo_id"]
        for capture in world.capture_log:
            if capture.photo_id == photo_id:
                if photo_id not in world.inspected_photo_ids:
                    world.inspected_photo_ids.append(photo_id)
                return capture.model_dump(mode="json") | {"quality_score": capture.quality_score}
        return {"error": f"unknown photo_id {photo_id}"}

    def estimate_view(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        sensor: SensorType = args.get("sensor", "thermal")
        estimate = estimate_visible_targets(
            telemetry=world.telemetry,
            targets=world.assets,
            sensor=sensor,
            weather=world.weather,
            hidden_defects=[],
        )
        return estimate.model_dump(mode="json") | {"quality_score": estimate.quality_score}

    def _capture(self, world: EpisodeWorld, sensor: SensorType, args: dict[str, Any]) -> dict[str, Any]:
        world.telemetry.camera.active_source = sensor
        capture = self._controller.capture_image(world, sensor, args.get("label"))
        self._update_checklist_from_capture(world, capture)
        return capture.model_dump(mode="json") | {"quality_score": capture.quality_score}

    def _update_checklist_from_capture(self, world: EpisodeWorld, capture) -> None:
        if capture.sensor == "thermal":
            covered = set(world.checklist_status.thermal_rows_covered)
            for target_id in capture.targets_visible:
                if target_id in world.mission.required_rows and capture.quality_score >= 0.55:
                    covered.add(target_id)
            world.checklist_status.thermal_rows_covered = sorted(covered)
            anomalies = set(world.checklist_status.anomalies_detected)
            anomalies.update(capture.detected_anomalies)
            world.checklist_status.anomalies_detected = sorted(anomalies)
        if capture.sensor == "rgb":
            for anomaly in world.checklist_status.anomalies_detected:
                defect = next((item for item in world.hidden_defects if item.defect_id == anomaly), None)
                target_visible = defect is not None and defect.target_id in capture.targets_visible
                if target_visible and capture.quality_score >= 0.55:
                    world.checklist_status.anomaly_rgb_pairs.setdefault(anomaly, capture.photo_id)
