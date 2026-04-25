"""Controller abstraction for all drone backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dronecaptureops.core.models import Capture, Pose, SensorType, Telemetry
from dronecaptureops.core.state import EpisodeWorld


class DroneController(ABC):
    """Backend-neutral interface used by the core environment."""

    @abstractmethod
    def reset(self, world: EpisodeWorld) -> None:
        """Reset backend state for a new episode."""

    @abstractmethod
    def get_telemetry(self, world: EpisodeWorld) -> Telemetry:
        """Return visible drone telemetry."""

    @abstractmethod
    def takeoff(self, world: EpisodeWorld, altitude_m: float) -> dict:
        """Take off to a target altitude."""

    @abstractmethod
    def fly_to(self, world: EpisodeWorld, pose: Pose, speed_mps: float) -> dict:
        """Fly to a safe waypoint."""

    @abstractmethod
    def hover(self, world: EpisodeWorld, seconds: float) -> dict:
        """Hover in place."""

    @abstractmethod
    def set_gimbal(self, world: EpisodeWorld, pitch_deg: float, yaw_deg: float | None = None) -> dict:
        """Set camera gimbal orientation."""

    @abstractmethod
    def capture_image(self, world: EpisodeWorld, sensor: SensorType, label: str | None = None) -> Capture:
        """Capture an RGB or thermal image."""

    @abstractmethod
    def return_home(self, world: EpisodeWorld) -> dict:
        """Return to home pose."""

    @abstractmethod
    def land(self, world: EpisodeWorld) -> dict:
        """Land the drone."""
