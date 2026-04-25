"""Typed data models shared across the environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator


SensorType = Literal["rgb", "thermal"]


class Pose(BaseModel):
    """Drone pose in a local ENU-like metric frame."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0


class GimbalState(BaseModel):
    """Camera gimbal state in degrees."""

    pitch_deg: float = -45.0
    yaw_deg: float = 0.0


class Telemetry(BaseModel):
    """Visible drone telemetry."""

    pose: Pose
    gimbal: GimbalState
    battery_pct: float = 100.0
    in_air: bool = False
    landed: bool = True
    mode: str = "idle"


class RectZone(BaseModel):
    """Axis-aligned rectangular zone in local map coordinates."""

    zone_id: str
    label: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    zone_type: Literal["no_fly", "privacy", "obstacle"] = "no_fly"


class TargetSurface(BaseModel):
    """Inspectable surface such as a solar row."""

    target_id: str
    label: str
    center_x: float
    center_y: float
    center_z: float = 0.0
    width_m: float = 10.0
    height_m: float = 3.0
    normal_yaw_deg: float = 180.0


class HiddenDefect(BaseModel):
    """Verifier-only defect state. Never expose this in observations."""

    defect_id: str
    target_id: str
    defect_type: str
    severity: float = Field(ge=0.0, le=1.0)


class WeatherState(BaseModel):
    """Simple weather model affecting capture quality."""

    wind_mps: float = 2.0
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)


class MissionChecklist(BaseModel):
    """Mission requirements that are visible to the agent."""

    mission_id: str
    instruction: str
    required_rows: list[str]
    thermal_overview_required: bool = True
    rgb_closeup_for_anomalies: bool = True
    must_return_home: bool = True
    min_battery_at_done_pct: float = 20.0


class Capture(BaseModel):
    """Structured capture metadata returned by camera tools."""

    photo_id: str
    sensor: SensorType
    label: str | None = None
    pose: Pose
    gimbal: GimbalState
    targets_visible: list[str] = Field(default_factory=list)
    coverage_pct: float = 0.0
    occlusion_pct: float = 0.0
    resolution_score: float = 0.0
    view_angle_score: float = 0.0
    blur_score: float = 0.0
    detected_anomalies: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def quality_score(self) -> float:
        """Aggregate quality score used by reward components."""

        return round(
            0.30 * self.coverage_pct
            + 0.25 * self.resolution_score
            + 0.20 * self.view_angle_score
            + 0.15 * self.blur_score
            + 0.10 * (1.0 - self.occlusion_pct),
            4,
        )


class EvidenceReport(BaseModel):
    """Final evidence pack submitted by the agent."""

    summary: str
    photo_ids: list[str] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)


class ChecklistStatus(BaseModel):
    """Visible progress toward mission completion."""

    thermal_rows_covered: list[str] = Field(default_factory=list)
    anomalies_detected: list[str] = Field(default_factory=list)
    anomaly_rgb_pairs: dict[str, str] = Field(default_factory=dict)
    returned_home: bool = False
    landed: bool = False
    evidence_submitted: bool = False
    complete: bool = False


class RewardBreakdown(BaseModel):
    """Logged reward components for training and debugging."""

    format_validity: float = 0.0
    flight_success: float = 0.0
    target_coverage: float = 0.0
    capture_quality: float = 0.0
    defect_visibility: float = 0.0
    checklist_completion: float = 0.0
    route_efficiency: float = 0.0
    battery_management: float = 0.0
    safety_compliance: float = 1.0
    report_grounding: float = 0.0
    recovery_behavior: float = 0.0
    penalties: float = 0.0
    safety_gate: float = 1.0
    total: float = 0.0


class SiteMap(BaseModel):
    """Visible map data."""

    domain: str
    home: Pose
    targets: list[TargetSurface]
    restricted_zones: list[RectZone]


class RawDroneAction(Action):
    """Public action payload accepted by OpenEnv step()."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def tool_name_must_exist(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("tool_name is required")
        return value.strip()


class DroneObservation(Observation):
    """Visible observation returned to the agent."""

    system_message: str = ""
    error: str | None = None
    available_tools: list[str] = Field(default_factory=list)
    telemetry: Telemetry | None = None
    mission: MissionChecklist | None = None
    site_map: SiteMap | None = None
    last_capture: Capture | None = None
    capture_log: list[Capture] = Field(default_factory=list)
    checklist_status: ChecklistStatus = Field(default_factory=ChecklistStatus)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    action_result: dict[str, Any] = Field(default_factory=dict)


class DroneVisibleState(State):
    """Serializable visible state. Hidden defects and verifier labels are omitted."""

    domain: str = ""
    scenario_seed: int = 0
    telemetry: Telemetry | None = None
    checklist_status: ChecklistStatus = Field(default_factory=ChecklistStatus)
    captures_taken: int = 0
    safety_violations: list[str] = Field(default_factory=list)
    done: bool = False
