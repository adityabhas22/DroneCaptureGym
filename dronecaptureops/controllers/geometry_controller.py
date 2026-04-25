"""Fast deterministic controller backed by geometry simulation."""

from __future__ import annotations

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.core.models import Capture, GimbalState, Pose, SensorType, Telemetry
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.simulation import battery
from dronecaptureops.simulation.camera import estimate_visible_targets
from dronecaptureops.simulation.geometry import flight_distance
from dronecaptureops.simulation.world import mark_return_status


class GeometryController(DroneController):
    """Controller implementation used for fast RL training."""

    def reset(self, world: EpisodeWorld) -> None:
        world.telemetry = Telemetry(
            pose=world.home_pose.model_copy(deep=True),
            gimbal=GimbalState(),
            battery_pct=100.0,
            in_air=False,
            landed=True,
            mode="idle",
        )

    def get_telemetry(self, world: EpisodeWorld) -> Telemetry:
        return world.telemetry.model_copy(deep=True)

    def takeoff(self, world: EpisodeWorld, altitude_m: float) -> dict:
        pose = world.telemetry.pose.model_copy(deep=True)
        pose.z = altitude_m
        world.telemetry.pose = pose
        world.telemetry.in_air = True
        world.telemetry.landed = False
        world.telemetry.mode = "guided"
        world.telemetry.battery_pct = battery.drain_for_takeoff(world.telemetry.battery_pct)
        mark_return_status(world)
        return {"altitude_m": altitude_m}

    def fly_to(self, world: EpisodeWorld, pose: Pose, speed_mps: float) -> dict:
        start = world.telemetry.pose.model_copy(deep=True)
        distance_m = flight_distance(start, pose)
        world.telemetry.pose = pose.model_copy(deep=True)
        world.telemetry.in_air = True
        world.telemetry.landed = False
        world.telemetry.mode = "guided"
        world.telemetry.battery_pct = battery.drain_for_flight(world.telemetry.battery_pct, distance_m)
        world.distance_flown_m += distance_m
        mark_return_status(world)
        return {"distance_m": round(distance_m, 3), "speed_mps": speed_mps}

    def hover(self, world: EpisodeWorld, seconds: float) -> dict:
        world.telemetry.mode = "hover"
        world.telemetry.battery_pct = battery.drain_for_hover(world.telemetry.battery_pct, seconds)
        mark_return_status(world)
        return {"seconds": seconds}

    def set_gimbal(self, world: EpisodeWorld, pitch_deg: float, yaw_deg: float | None = None) -> dict:
        world.telemetry.gimbal.pitch_deg = pitch_deg
        if yaw_deg is not None:
            world.telemetry.gimbal.yaw_deg = yaw_deg
        return world.telemetry.gimbal.model_dump()

    def capture_image(self, world: EpisodeWorld, sensor: SensorType, label: str | None = None) -> Capture:
        photo_id = f"IMG-{'T' if sensor == 'thermal' else 'R'}-{len(world.capture_log) + 1:03d}"
        capture = estimate_visible_targets(
            telemetry=world.telemetry,
            targets=world.targets,
            sensor=sensor,
            weather=world.weather,
            hidden_defects=world.hidden_defects,
            photo_id=photo_id,
            label=label,
        )
        world.capture_log.append(capture)
        world.telemetry.battery_pct = battery.drain_for_capture(world.telemetry.battery_pct)
        return capture

    def return_home(self, world: EpisodeWorld) -> dict:
        target = world.home_pose.model_copy(deep=True)
        target.z = max(world.telemetry.pose.z, 10.0)
        result = self.fly_to(world, target, speed_mps=5.0)
        world.telemetry.mode = "return_home"
        mark_return_status(world)
        return result

    def land(self, world: EpisodeWorld) -> dict:
        pose = world.telemetry.pose.model_copy(deep=True)
        pose.z = 0.0
        world.telemetry.pose = pose
        world.telemetry.in_air = False
        world.telemetry.landed = True
        world.telemetry.mode = "landed"
        world.telemetry.battery_pct = battery.drain_for_land(world.telemetry.battery_pct)
        mark_return_status(world)
        return {"landed": True}
