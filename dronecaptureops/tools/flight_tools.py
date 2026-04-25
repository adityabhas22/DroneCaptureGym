"""Flight tool handlers."""

from __future__ import annotations

from typing import Any

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.core.models import Pose
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.simulation.safety import SafetyChecker


class FlightTools:
    """High-level flight tools backed by a DroneController."""

    def __init__(self, controller: DroneController, safety: SafetyChecker) -> None:
        self._controller = controller
        self._safety = safety

    def takeoff(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        altitude_m = float(args["altitude_m"])
        self._safety.validate_takeoff(world, altitude_m)
        return self._controller.takeoff(world, altitude_m)

    def fly_to_viewpoint(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        target = Pose(
            x=float(args["x"]),
            y=float(args["y"]),
            z=float(args["z"]),
            yaw_deg=float(args.get("yaw_deg", world.telemetry.pose.yaw_deg)),
        )
        speed = float(args.get("speed_mps", 5.0))
        self._safety.validate_waypoint(world, target, speed)
        return self._controller.fly_to(world, target, speed)

    def hover(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        seconds = max(0.0, min(float(args.get("seconds", 1.0)), 60.0))
        return self._controller.hover(world, seconds)

    def return_home(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        target = world.home_pose.model_copy(deep=True)
        target.z = max(world.telemetry.pose.z, 10.0)
        self._safety.validate_waypoint(world, target, 5.0)
        return self._controller.return_home(world)

    def land(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        return self._controller.land(world)
