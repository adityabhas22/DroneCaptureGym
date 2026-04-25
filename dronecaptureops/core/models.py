"""Typed data models shared across the environment."""

from __future__ import annotations

from typing import Any, Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field, field_validator


SensorType = Literal["rgb", "thermal"]
AutopilotMode = Literal["idle", "guided", "hover", "return_home", "landed"]
CommandStatus = Literal["idle", "accepted", "rejected", "completed", "failed"]
AssetType = Literal["solar_row", "inverter", "combiner_box", "substation", "access_road"]
AirspaceZoneType = Literal["no_fly", "privacy", "obstacle"]
ConstraintLevel = Literal["soft", "hard"]
GimbalFrameMode = Literal["body", "earth", "roi"]
CameraSource = Literal["rgb", "thermal", "rgb_thermal"]
ScenarioDifficulty = Literal["easy", "medium", "hard"]


class Pose3D(BaseModel):
    """Drone pose in a local ENU-like metric frame."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0


class Pose(Pose3D):
    """Backward-compatible pose alias used by existing tools."""


class Velocity3D(BaseModel):
    """Drone velocity in local m/s components."""

    vx_mps: float = 0.0
    vy_mps: float = 0.0
    vz_mps: float = 0.0
    ground_speed_mps: float = 0.0
    air_speed_mps: float = 0.0


class Attitude(BaseModel):
    """Vehicle attitude in degrees."""

    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0


class AutopilotState(BaseModel):
    """DroneKit/ArduPilot-inspired autopilot status."""

    mode: AutopilotMode = "idle"
    armed: bool = False
    is_armable: bool = True
    system_status: str = "STANDBY"
    ekf_ok: bool = True
    last_heartbeat_s: float = 0.0
    command_status: CommandStatus = "idle"
    last_command: str | None = None


class GpsState(BaseModel):
    """Visible GPS quality state."""

    fix_type: int = 3
    satellites_visible: int = 12
    eph: float = 0.8
    epv: float = 1.1
    hdop: float = 0.9


class BatteryState(BaseModel):
    """Battery state mirroring common MAVLink/DroneKit fields."""

    voltage_v: float = 16.8
    current_a: float = 0.0
    level_pct: float = 100.0
    consumed_mah: float = 0.0
    reserve_pct: float = 20.0


class LinkState(BaseModel):
    """Command-and-control link status."""

    connected: bool = True
    rssi_pct: float = 100.0
    packet_loss_pct: float = 0.0
    latency_ms: int = 30


class RangefinderState(BaseModel):
    """Downward rangefinder state."""

    distance_m: float | None = None
    voltage_v: float | None = None
    healthy: bool = True


class GimbalState(BaseModel):
    """Camera gimbal state in degrees."""

    pitch_deg: float = -45.0
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    frame_mode: GimbalFrameMode = "body"
    target_asset_id: str | None = None
    roi: Pose3D | None = None
    ready: bool = True


class CameraState(BaseModel):
    """Visible camera payload state."""

    active_source: CameraSource = "rgb"
    supported_sources: list[CameraSource] = Field(default_factory=lambda: ["rgb", "thermal", "rgb_thermal"])
    zoom_level: float = 1.0
    focus_mode: Literal["auto", "manual"] = "auto"
    focus_distance_m: float | None = None
    capture_ready: bool = True
    storage_remaining: int = 256


class Telemetry(BaseModel):
    """Visible drone telemetry with DroneKit-style nested state."""

    pose: Pose
    velocity: Velocity3D = Field(default_factory=Velocity3D)
    attitude: Attitude = Field(default_factory=Attitude)
    autopilot: AutopilotState = Field(default_factory=AutopilotState)
    gps: GpsState = Field(default_factory=GpsState)
    battery: BatteryState = Field(default_factory=BatteryState)
    link: LinkState = Field(default_factory=LinkState)
    rangefinder: RangefinderState = Field(default_factory=RangefinderState)
    gimbal: GimbalState
    camera: CameraState = Field(default_factory=CameraState)
    weather_band: Literal["low", "mid", "high"] = "low"
    elapsed_time_s: float = 0.0
    distance_flown_m: float = 0.0
    battery_pct: float = 100.0
    in_air: bool = False
    landed: bool = True
    mode: str = "idle"

    def sync_legacy_fields(self) -> None:
        """Keep legacy flat fields aligned with nested telemetry."""

        self.mode = self.autopilot.mode
        self.battery_pct = self.battery.level_pct
        self.landed = self.autopilot.mode in {"idle", "landed"} and self.pose.z <= 0.0
        self.in_air = self.autopilot.armed and not self.landed
        self.attitude.yaw_deg = self.pose.yaw_deg
        self.rangefinder.distance_m = self.pose.z if self.pose.z > 0 else None


class AirspaceZone(BaseModel):
    """Axis-aligned operational constraint zone in local map coordinates."""

    zone_id: str
    label: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_altitude_m: float = 0.0
    max_altitude_m: float = 120.0
    zone_type: AirspaceZoneType = "no_fly"
    constraint_level: ConstraintLevel = "hard"
    reason: str = ""


class RectZone(AirspaceZone):
    """Backward-compatible rectangular zone alias."""


class AssetGeometry(BaseModel):
    """2.5D asset footprint and viewing normal."""

    center_x: float
    center_y: float
    center_z: float = 0.0
    width_m: float = 10.0
    height_m: float = 3.0
    normal_yaw_deg: float = 180.0
    tilt_deg: float = 0.0


class StandoffBand(BaseModel):
    """Safe distance band for a capture type."""

    name: Literal["far", "mid", "close"]
    min_m: float
    max_m: float
    preferred_m: float


class InspectableAsset(BaseModel):
    """Generic inspectable site entity."""

    asset_id: str
    asset_type: AssetType
    label: str
    geometry: AssetGeometry
    required_modalities: list[SensorType] = Field(default_factory=lambda: ["thermal"])
    safe_standoff_bands: list[StandoffBand] = Field(default_factory=list)
    visibility_tags: list[str] = Field(default_factory=list)
    public_notes: list[str] = Field(default_factory=list)

    @property
    def target_id(self) -> str:
        """Compatibility with the original TargetSurface API."""

        return self.asset_id

    @property
    def center_x(self) -> float:
        return self.geometry.center_x

    @property
    def center_y(self) -> float:
        return self.geometry.center_y

    @property
    def center_z(self) -> float:
        return self.geometry.center_z

    @property
    def width_m(self) -> float:
        return self.geometry.width_m

    @property
    def height_m(self) -> float:
        return self.geometry.height_m

    @property
    def normal_yaw_deg(self) -> float:
        return self.geometry.normal_yaw_deg


class TargetSurface(InspectableAsset):
    """Backward-compatible inspectable surface alias."""

    def __init__(self, **data: Any) -> None:
        if "asset_id" not in data and "target_id" in data:
            data["asset_id"] = data.pop("target_id")
        if "asset_type" not in data:
            data["asset_type"] = "solar_row"
        if "geometry" not in data:
            data["geometry"] = AssetGeometry(
                center_x=data.pop("center_x"),
                center_y=data.pop("center_y"),
                center_z=data.pop("center_z", 0.0),
                width_m=data.pop("width_m", 10.0),
                height_m=data.pop("height_m", 3.0),
                normal_yaw_deg=data.pop("normal_yaw_deg", 180.0),
            )
        super().__init__(**data)


class Viewpoint(BaseModel):
    """Named candidate capture pose for an asset or group of assets."""

    viewpoint_id: str
    label: str
    pose: Pose
    asset_ids: list[str] = Field(default_factory=list)
    standoff_bucket: Literal["far", "mid", "close"] = "mid"
    suitable_modalities: list[SensorType] = Field(default_factory=lambda: ["rgb", "thermal"])
    notes: list[str] = Field(default_factory=list)


class HiddenDefect(BaseModel):
    """Verifier-only defect state. Never expose this in observations.

    Thresholds are tuned to per-target quality (0..1 from the 3D-FOV camera
    sim). A defect is captured when (a) the simulator flags it in
    detected_anomalies, and (b) the underlying target's per-target quality
    clears `min_quality` in a thermal capture.
    """

    defect_id: str
    target_id: str
    defect_type: str
    severity: float = Field(ge=0.0, le=1.0)
    required_sensor: SensorType = "thermal"
    requires_rgb_context: bool = True
    min_quality: float = 0.55
    min_resolution_score: float = 0.40
    max_occlusion: float = 0.30
    max_view_angle_deg: float = 65.0
    weight: float = 2.0
    counts_for_issue_reward: bool = True


class WeatherState(BaseModel):
    """Simple weather model affecting capture quality."""

    wind_mps: float = 2.0
    visibility: float = Field(default=1.0, ge=0.0, le=1.0)
    irradiance_wm2: float = 760.0
    cloud_cover_oktas: int = Field(default=1, ge=0, le=8)
    ambient_temp_c: float = 28.0

    @property
    def wind_band(self) -> Literal["low", "mid", "high"]:
        if self.wind_mps < 3.0:
            return "low"
        if self.wind_mps < 5.5:
            return "mid"
        return "high"


class MissionChecklist(BaseModel):
    """Mission requirements that are visible to the agent."""

    mission_id: str
    instruction: str
    required_rows: list[str]
    thermal_overview_required: bool = True
    rgb_closeup_for_anomalies: bool = True
    must_return_home: bool = True
    min_battery_at_done_pct: float = 20.0
    scenario_family: str = "baseline_hotspot"
    difficulty: ScenarioDifficulty = "easy"
    environmental_constraints: list[str] = Field(default_factory=list)


class Capture(BaseModel):
    """Structured capture metadata returned by camera tools."""

    photo_id: str
    sensor: SensorType
    label: str | None = None
    pose: Pose
    gimbal: GimbalState
    camera: CameraState | None = None
    asset_ids: list[str] = Field(default_factory=list)
    targets_visible: list[str] = Field(default_factory=list)
    viewpoint_id: str | None = None
    coverage_pct: float = 0.0
    occlusion_pct: float = 0.0
    resolution_score: float = 0.0
    view_angle_score: float = 0.0
    blur_score: float = 0.0
    standoff_score: float = 0.0
    gsd_score: float = 0.0
    glare_score: float = 1.0
    thermal_contrast_score: float = 0.0
    detected_anomalies: list[str] = Field(default_factory=list)
    quality_inputs: dict[str, float] = Field(default_factory=dict)
    per_target_quality: dict[str, float] = Field(default_factory=dict)
    per_target_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    def target_quality(self, target_id: str) -> float:
        """Return per-target quality if recorded, otherwise fall back to aggregate."""

        if target_id in self.per_target_quality:
            return self.per_target_quality[target_id]
        return self.quality_score if target_id in self.targets_visible else 0.0

    @property
    def quality_score(self) -> float:
        """Aggregate quality score used by reward components.

        Prefers per-target quality (mean over visible targets) when populated;
        otherwise falls back to the legacy aggregate scoring.
        """

        if self.per_target_quality:
            scores = list(self.per_target_quality.values())
            return round(sum(scores) / len(scores), 4)
        return round(
            0.30 * self.coverage_pct
            + 0.25 * self.resolution_score
            + 0.20 * self.view_angle_score
            + 0.15 * self.blur_score
            + 0.10 * (1.0 - self.occlusion_pct),
            4,
        )


class InspectionArtifact(Capture):
    """Alias for inspection evidence artifacts."""


class EvidenceReport(BaseModel):
    """Final evidence pack submitted by the agent."""

    summary: str = ""
    photo_ids: list[str] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    mission_status: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    issues_found: list[dict[str, Any]] = Field(default_factory=list)
    open_items: list[dict[str, Any] | str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)


class ChecklistStatus(BaseModel):
    """Visible progress toward mission completion."""

    thermal_rows_covered: list[str] = Field(default_factory=list)
    anomalies_detected: list[str] = Field(default_factory=list)
    anomaly_rgb_pairs: dict[str, str] = Field(default_factory=dict)
    returned_home: bool = False
    landed: bool = False
    evidence_submitted: bool = False
    complete: bool = False


class InspectionAffordances(BaseModel):
    """Visible workflow guidance for the inspection director."""

    mission_phase: str = "preflight"
    waiting_on: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    next_due_steps: int | None = None
    recommended_action_categories: list[str] = Field(default_factory=list)
    action_availability: dict[str, bool] = Field(default_factory=dict)
    pending_asset_ids: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)


class RewardBreakdown(BaseModel):
    """Logged reward components for training and debugging."""

    format_validity: float = 0.0
    flight_success: float = 0.0
    evidence_success: float = 0.0
    required_coverage: float = 0.0
    issue_capture: float = 0.0
    operational_efficiency: float = 0.0
    grounded_report: float = 0.0
    process_reward: float = 0.0
    integrity_gate: float = 1.0
    value_per_photo: float = 0.0
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
    debug: dict[str, Any] = Field(default_factory=dict)


class SiteMap(BaseModel):
    """Visible map data."""

    domain: str
    home: Pose
    assets: list[InspectableAsset] = Field(default_factory=list)
    airspace_zones: list[AirspaceZone] = Field(default_factory=list)
    viewpoints: list[Viewpoint] = Field(default_factory=list)
    targets: list[InspectableAsset] = Field(default_factory=list)
    restricted_zones: list[AirspaceZone] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.targets:
            self.targets = list(self.assets)
        if not self.restricted_zones:
            self.restricted_zones = list(self.airspace_zones)


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
    visible_assets: list[InspectableAsset] = Field(default_factory=list)
    evidence_artifacts: list[InspectionArtifact] = Field(default_factory=list)
    inspection_affordances: InspectionAffordances = Field(default_factory=InspectionAffordances)
    tool_catalog: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    state_summary: dict[str, Any] = Field(default_factory=dict)
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
    visible_assets: list[InspectableAsset] = Field(default_factory=list)
    evidence_artifacts: list[InspectionArtifact] = Field(default_factory=list)
    checklist_status: ChecklistStatus = Field(default_factory=ChecklistStatus)
    captures_taken: int = 0
    safety_violations: list[str] = Field(default_factory=list)
    done: bool = False
