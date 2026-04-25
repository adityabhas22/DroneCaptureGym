"""Flight tool handlers."""

from __future__ import annotations

from typing import Any

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.core.errors import ActionValidationError
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

    def move_to_asset(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        asset_id = args["asset_id"]
        standoff_bucket = args.get("standoff_bucket", "mid")
        asset = next((item for item in world.assets if item.asset_id == asset_id), None)
        if asset is None:
            raise ActionValidationError(f"unknown asset_id: {asset_id}")
        viewpoint = next(
            (
                candidate
                for candidate in world.viewpoints
                if asset_id in candidate.asset_ids and candidate.standoff_bucket == standoff_bucket
            ),
            None,
        )
        if viewpoint is not None:
            target = viewpoint.pose.model_copy(deep=True)
            viewpoint_id = viewpoint.viewpoint_id
        else:
            band = next((item for item in asset.safe_standoff_bands if item.name == standoff_bucket), None)
            standoff_m = band.preferred_m if band else 22.0
            target = Pose(
                x=asset.center_x,
                y=asset.center_y + standoff_m,
                z=18.0 if standoff_bucket == "far" else 12.0,
                yaw_deg=-90.0,
            )
            viewpoint_id = None
        speed = float(args.get("speed_mps", 5.0))
        self._safety.validate_waypoint(world, target, speed)
        result = self._controller.fly_to(world, target, speed)
        result.update(
            {
                "asset_id": asset_id,
                "standoff_bucket": standoff_bucket,
                "viewpoint_id": viewpoint_id,
            }
        )
        return result
