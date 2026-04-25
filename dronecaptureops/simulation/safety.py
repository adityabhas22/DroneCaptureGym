"""Safety validation for high-level drone actions."""

from __future__ import annotations

from dronecaptureops.core.constants import MAX_ALTITUDE_M, MAX_SPEED_MPS, MIN_ALTITUDE_M
from dronecaptureops.core.errors import SafetyViolationError
from dronecaptureops.core.models import Pose
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.utils.geo_utils import pose_in_zone, segment_intersects_rect


class SafetyChecker:
    """Checks safety constraints before controller execution."""

    def validate_takeoff(self, world: EpisodeWorld, altitude_m: float) -> None:
        if altitude_m < MIN_ALTITUDE_M or altitude_m > MAX_ALTITUDE_M:
            raise SafetyViolationError(f"unsafe_altitude:{altitude_m}")
        if world.telemetry.battery.level_pct <= 1.0:
            raise SafetyViolationError("battery_exhausted")

    def validate_waypoint(self, world: EpisodeWorld, target: Pose, speed_mps: float) -> None:
        if target.z < MIN_ALTITUDE_M or target.z > MAX_ALTITUDE_M:
            raise SafetyViolationError(f"unsafe_altitude:{target.z}")
        if speed_mps <= 0 or speed_mps > MAX_SPEED_MPS:
            raise SafetyViolationError(f"unsafe_speed:{speed_mps}")
        for zone in world.airspace_zones:
            if zone.zone_type in {"no_fly", "obstacle"}:
                if target.z < zone.min_altitude_m or target.z > zone.max_altitude_m:
                    continue
                if pose_in_zone(target, zone) or segment_intersects_rect(world.telemetry.pose, target, zone):
                    raise SafetyViolationError(f"{zone.zone_type}_violation:{zone.zone_id}")
        if world.telemetry.battery.level_pct <= 1.0:
            raise SafetyViolationError("battery_exhausted")

    def validate_gimbal(self, pitch_deg: float, yaw_deg: float | None) -> None:
        if pitch_deg < -90.0 or pitch_deg > 20.0:
            raise SafetyViolationError(f"invalid_gimbal_pitch:{pitch_deg}")
        if yaw_deg is not None and (yaw_deg < -180.0 or yaw_deg > 180.0):
            raise SafetyViolationError(f"invalid_gimbal_yaw:{yaw_deg}")

    def validate_capture(self, world: EpisodeWorld) -> None:
        """Block imagery from privacy-sensitive capture zones."""

        pose = world.telemetry.pose
        for zone in world.airspace_zones:
            if zone.zone_type != "privacy":
                continue
            if pose.z < zone.min_altitude_m or pose.z > zone.max_altitude_m:
                continue
            if pose_in_zone(pose, zone):
                raise SafetyViolationError(f"privacy_capture_violation:{zone.zone_id}")
