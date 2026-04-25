"""Future DroneKit / ArduPilot SITL adapter."""

from __future__ import annotations

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.core.models import Capture, Pose, SensorType, Telemetry
from dronecaptureops.core.state import EpisodeWorld


class DroneKitSITLController(DroneController):
    """Placeholder adapter preserving the controller boundary for SITL work."""

    def __init__(self) -> None:
        self._message = "DroneKit SITL backend is not implemented yet; use GeometryController for MVP."

    def _not_ready(self) -> None:
        raise NotImplementedError(self._message)

    def reset(self, world: EpisodeWorld) -> None:
        self._not_ready()

    def get_telemetry(self, world: EpisodeWorld) -> Telemetry:
        self._not_ready()

    def takeoff(self, world: EpisodeWorld, altitude_m: float) -> dict:
        self._not_ready()

    def fly_to(self, world: EpisodeWorld, pose: Pose, speed_mps: float) -> dict:
        self._not_ready()

    def hover(self, world: EpisodeWorld, seconds: float) -> dict:
        self._not_ready()

    def set_gimbal(self, world: EpisodeWorld, pitch_deg: float, yaw_deg: float | None = None) -> dict:
        self._not_ready()

    def capture_image(self, world: EpisodeWorld, sensor: SensorType, label: str | None = None) -> Capture:
        self._not_ready()

    def return_home(self, world: EpisodeWorld) -> dict:
        self._not_ready()

    def land(self, world: EpisodeWorld) -> dict:
        self._not_ready()
