"""Fast deterministic controller backed by geometry simulation."""

from __future__ import annotations

import math

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.core.models import Capture, GimbalState, InspectionArtifact, Pose, SensorType, Telemetry
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.simulation import battery
from dronecaptureops.simulation.camera import estimate_visible_targets
from dronecaptureops.simulation.geometry import flight_distance
from dronecaptureops.simulation.world import mark_return_status


class GeometryController(DroneController):
    """Controller implementation used for fast RL training."""

    def reset(self, world: EpisodeWorld) -> None:
        # Preserve scenario-set fields (battery, weather band) that the
        # builder may have overridden for a task-conditioned mission. The
        # controller is responsible for putting the drone in a pre-takeoff
        # pose, not for re-randomizing battery or weather.
        scenario_battery = world.telemetry.battery.model_copy(deep=True)
        scenario_weather_band = world.telemetry.weather_band
        world.telemetry = Telemetry(
            pose=world.home_pose.model_copy(deep=True),
            gimbal=GimbalState(),
            battery=scenario_battery,
            battery_pct=scenario_battery.level_pct,
            weather_band=scenario_weather_band,
            in_air=False,
            landed=True,
            mode="idle",
        )
        world.telemetry.autopilot.mode = "idle"
        world.telemetry.autopilot.armed = False
        world.telemetry.autopilot.is_armable = True
        world.telemetry.autopilot.system_status = "STANDBY"
        world.telemetry.autopilot.ekf_ok = True
        world.telemetry.autopilot.command_status = "completed"
        world.telemetry.autopilot.last_command = "reset"
        world.telemetry.weather_band = world.weather.wind_band
        self._sync_telemetry(world, current_a=0.0)

    def get_telemetry(self, world: EpisodeWorld) -> Telemetry:
        return world.telemetry.model_copy(deep=True)

    def takeoff(self, world: EpisodeWorld, altitude_m: float) -> dict:
        self._mark_command(world, "takeoff", "accepted")
        pose = world.telemetry.pose.model_copy(deep=True)
        pose.z = altitude_m
        world.telemetry.pose = pose
        world.telemetry.autopilot.armed = True
        world.telemetry.autopilot.mode = "guided"
        world.telemetry.autopilot.system_status = "ACTIVE"
        self._set_battery_pct(world, battery.drain_for_takeoff(world.telemetry.battery.level_pct), current_a=8.5)
        self._advance_time(world, 8.0)
        self._mark_command(world, "takeoff", "completed")
        mark_return_status(world)
        return {"altitude_m": altitude_m, "command_status": world.telemetry.autopilot.command_status}

    def fly_to(self, world: EpisodeWorld, pose: Pose, speed_mps: float) -> dict:
        self._mark_command(world, "fly_to", "accepted")
        start = world.telemetry.pose.model_copy(deep=True)
        distance_m = flight_distance(start, pose)
        duration_s = distance_m / max(speed_mps, 0.1)
        world.telemetry.pose = pose.model_copy(deep=True)
        world.telemetry.autopilot.armed = True
        world.telemetry.autopilot.mode = "guided"
        world.telemetry.autopilot.system_status = "ACTIVE"
        world.telemetry.velocity.vx_mps = (pose.x - start.x) / duration_s if duration_s else 0.0
        world.telemetry.velocity.vy_mps = (pose.y - start.y) / duration_s if duration_s else 0.0
        world.telemetry.velocity.vz_mps = (pose.z - start.z) / duration_s if duration_s else 0.0
        world.telemetry.velocity.ground_speed_mps = min(speed_mps, math.hypot(world.telemetry.velocity.vx_mps, world.telemetry.velocity.vy_mps))
        world.telemetry.velocity.air_speed_mps = min(speed_mps + max(0.0, world.weather.wind_mps * 0.15), speed_mps + 2.0)
        self._set_battery_pct(world, battery.drain_for_flight(world.telemetry.battery.level_pct, distance_m), current_a=10.0)
        self._advance_time(world, duration_s)
        world.distance_flown_m += distance_m
        world.telemetry.distance_flown_m = world.distance_flown_m
        self._mark_command(world, "fly_to", "completed")
        mark_return_status(world)
        return {"distance_m": round(distance_m, 3), "duration_s": round(duration_s, 2), "speed_mps": speed_mps}

    def hover(self, world: EpisodeWorld, seconds: float) -> dict:
        self._mark_command(world, "hover", "accepted")
        world.telemetry.autopilot.mode = "hover"
        world.telemetry.velocity.vx_mps = 0.0
        world.telemetry.velocity.vy_mps = 0.0
        world.telemetry.velocity.vz_mps = 0.0
        world.telemetry.velocity.ground_speed_mps = 0.0
        self._set_battery_pct(world, battery.drain_for_hover(world.telemetry.battery.level_pct, seconds), current_a=5.0)
        self._advance_time(world, seconds)
        self._mark_command(world, "hover", "completed")
        mark_return_status(world)
        return {"seconds": seconds}

    def set_gimbal(self, world: EpisodeWorld, pitch_deg: float, yaw_deg: float | None = None) -> dict:
        self._mark_command(world, "set_gimbal", "accepted")
        world.telemetry.gimbal.pitch_deg = pitch_deg
        if yaw_deg is not None:
            world.telemetry.gimbal.yaw_deg = yaw_deg
        world.telemetry.gimbal.frame_mode = "body"
        world.telemetry.gimbal.target_asset_id = None
        self._advance_time(world, 1.0)
        self._mark_command(world, "set_gimbal", "completed")
        return world.telemetry.gimbal.model_dump()

    def capture_image(self, world: EpisodeWorld, sensor: SensorType, label: str | None = None) -> Capture:
        self._mark_command(world, f"capture_{sensor}", "accepted")
        world.telemetry.camera.active_source = sensor
        photo_id = f"IMG-{'T' if sensor == 'thermal' else 'R'}-{len(world.capture_log) + 1:03d}"
        capture = estimate_visible_targets(
            telemetry=world.telemetry,
            targets=world.assets,
            sensor=sensor,
            weather=world.weather,
            hidden_defects=world.hidden_defects,
            photo_id=photo_id,
            label=label,
        )
        world.capture_log.append(capture)
        world.evidence_artifacts.append(InspectionArtifact.model_validate(capture.model_dump()))
        world.telemetry.camera.storage_remaining = max(0, world.telemetry.camera.storage_remaining - 1)
        self._set_battery_pct(world, battery.drain_for_capture(world.telemetry.battery.level_pct), current_a=6.0)
        self._advance_time(world, 2.0)
        self._mark_command(world, f"capture_{sensor}", "completed")
        return capture

    def return_home(self, world: EpisodeWorld) -> dict:
        self._mark_command(world, "return_home", "accepted")
        target = world.home_pose.model_copy(deep=True)
        target.z = max(world.telemetry.pose.z, 10.0)
        result = self.fly_to(world, target, speed_mps=5.0)
        world.telemetry.autopilot.mode = "return_home"
        self._mark_command(world, "return_home", "completed")
        mark_return_status(world)
        return result

    def land(self, world: EpisodeWorld) -> dict:
        self._mark_command(world, "land", "accepted")
        pose = world.telemetry.pose.model_copy(deep=True)
        pose.z = 0.0
        world.telemetry.pose = pose
        world.telemetry.autopilot.mode = "landed"
        world.telemetry.autopilot.armed = False
        world.telemetry.autopilot.system_status = "STANDBY"
        self._set_battery_pct(world, battery.drain_for_land(world.telemetry.battery.level_pct), current_a=1.0)
        self._advance_time(world, 6.0)
        self._mark_command(world, "land", "completed")
        mark_return_status(world)
        return {"landed": True}

    def _set_battery_pct(self, world: EpisodeWorld, level_pct: float, current_a: float) -> None:
        world.telemetry.battery.level_pct = round(level_pct, 3)
        world.telemetry.battery.voltage_v = round(13.6 + 3.2 * (level_pct / 100.0), 3)
        world.telemetry.battery.current_a = current_a
        world.telemetry.battery.consumed_mah = round((100.0 - level_pct) * 42.0, 3)
        self._sync_telemetry(world, current_a=current_a)

    def _advance_time(self, world: EpisodeWorld, seconds: float) -> None:
        world.elapsed_time_s += seconds
        world.telemetry.elapsed_time_s = round(world.elapsed_time_s, 3)
        world.telemetry.autopilot.last_heartbeat_s = 0.0
        self._sync_telemetry(world, current_a=world.telemetry.battery.current_a)

    def _mark_command(self, world: EpisodeWorld, command: str, status: str) -> None:
        world.telemetry.autopilot.last_command = command
        world.telemetry.autopilot.command_status = status
        self._sync_telemetry(world, current_a=world.telemetry.battery.current_a)

    def _sync_telemetry(self, world: EpisodeWorld, current_a: float) -> None:
        world.telemetry.battery.current_a = current_a
        world.telemetry.weather_band = world.weather.wind_band
        world.telemetry.sync_legacy_fields()
