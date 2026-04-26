"""Renderer-friendly scene and event serialization.

This module intentionally builds scenes from the same public objects exposed to
agents. It must not dump ``EpisodeWorld`` directly because that model contains
verifier-only state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from dronecaptureops.core.constants import (
    RGB_EFFECTIVE_RANGE_M,
    RGB_HFOV_DEG,
    RGB_VFOV_DEG,
    THERMAL_EFFECTIVE_RANGE_M,
    THERMAL_HFOV_DEG,
    THERMAL_VFOV_DEG,
)
from dronecaptureops.core.models import (
    AirspaceZone,
    Capture,
    ChecklistStatus,
    DroneObservation,
    GimbalState,
    InspectableAsset,
    MissionChecklist,
    Pose,
    RewardBreakdown,
    SiteMap,
    Telemetry,
    Viewpoint,
)
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.utils.serialization import to_jsonable

SCENE_SCHEMA_VERSION = "rich_sim.scene.v1"

RouteHistoryInput = Sequence[Pose | Mapping[str, Any] | Sequence[float] | BaseModel]


class ScenePose(BaseModel):
    """Pose in the local ENU-like metric frame used by the simulator."""

    x: float
    y: float
    z: float
    yaw_deg: float = 0.0


class ScenePoint2D(BaseModel):
    x: float
    y: float


class SceneMetadata(BaseModel):
    schema_version: str = SCENE_SCHEMA_VERSION
    episode_id: str | None = None
    domain: str | None = None
    scenario_family: str | None = None
    scenario_seed: int | None = None
    step_count: int | None = None
    done: bool = False


class SceneRoutePoint(BaseModel):
    pose: ScenePose
    elapsed_time_s: float | None = None
    label: str | None = None
    source: str | None = None


class SceneGimbal(BaseModel):
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    frame_mode: str
    target_asset_id: str | None = None
    roi: ScenePose | None = None


class SceneDrone(BaseModel):
    pose: ScenePose
    gimbal: SceneGimbal
    camera_source: str
    mode: str
    armed: bool
    in_air: bool
    landed: bool


class SceneTelemetrySummary(BaseModel):
    battery_pct: float
    voltage_v: float
    current_a: float
    consumed_mah: float
    reserve_pct: float
    ground_speed_mps: float
    air_speed_mps: float
    elapsed_time_s: float
    distance_flown_m: float
    weather_band: str
    gps_fix_type: int
    satellites_visible: int
    link_rssi_pct: float
    packet_loss_pct: float
    command_status: str
    last_command: str | None = None
    storage_remaining: int


class SceneHomePad(BaseModel):
    pose: ScenePose
    radius_m: float = 1.0
    label: str = "Home pad"


class SceneStandoffBand(BaseModel):
    name: str
    min_m: float
    max_m: float
    preferred_m: float


class SceneAsset(BaseModel):
    asset_id: str
    asset_type: str
    label: str
    center: ScenePose
    width_m: float
    height_m: float
    normal_yaw_deg: float
    tilt_deg: float
    required_modalities: list[str] = Field(default_factory=list)
    safe_standoff_bands: list[SceneStandoffBand] = Field(default_factory=list)
    visibility_tags: list[str] = Field(default_factory=list)
    public_notes: list[str] = Field(default_factory=list)


class SceneAirspaceZone(BaseModel):
    zone_id: str
    label: str
    zone_type: str
    constraint_level: str
    reason: str
    min_altitude_m: float
    max_altitude_m: float
    polygon_xy: list[ScenePoint2D]


class SceneViewpoint(BaseModel):
    viewpoint_id: str
    label: str
    pose: ScenePose
    asset_ids: list[str] = Field(default_factory=list)
    standoff_bucket: str
    suitable_modalities: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class SceneFrustum(BaseModel):
    sensor: str
    origin: ScenePose
    yaw_deg: float
    pitch_deg: float
    hfov_deg: float
    vfov_deg: float
    range_m: float


class SceneCapturePoint(BaseModel):
    photo_id: str
    sensor: str
    label: str | None = None
    pose: ScenePose
    gimbal: SceneGimbal
    frustum: SceneFrustum
    asset_ids: list[str] = Field(default_factory=list)
    targets_visible: list[str] = Field(default_factory=list)
    coverage_pct: float
    quality_score: float
    resolution_score: float
    view_angle_score: float
    blur_score: float
    standoff_score: float
    gsd_score: float
    glare_score: float
    thermal_contrast_score: float
    detected_anomalies: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SceneChecklistSummary(BaseModel):
    mission_id: str | None = None
    instruction: str | None = None
    required_rows: list[str] = Field(default_factory=list)
    thermal_rows_covered: list[str] = Field(default_factory=list)
    anomalies_detected: list[str] = Field(default_factory=list)
    anomaly_rgb_pairs: dict[str, str] = Field(default_factory=dict)
    anomaly_targets: dict[str, str] = Field(default_factory=dict)
    returned_home: bool = False
    landed: bool = False
    evidence_submitted: bool = False
    complete: bool = False
    mission_phase: str | None = None
    blockers: list[str] = Field(default_factory=list)
    pending_asset_ids: list[str] = Field(default_factory=list)
    suggested_tools: list[str] = Field(default_factory=list)
    next_due_steps: int | None = None


class SceneRewardSummary(BaseModel):
    total: float
    evidence_success: float
    required_coverage: float
    issue_capture: float
    operational_efficiency: float
    grounded_report: float
    process_reward: float
    capture_quality: float
    checklist_completion: float
    safety_compliance: float
    battery_management: float
    safety_gate: float
    integrity_gate: float
    penalties: float


class RichSimScene(BaseModel):
    """Complete renderer-facing scene graph with visible state only."""

    schema_version: str = SCENE_SCHEMA_VERSION
    metadata: SceneMetadata
    drone: SceneDrone | None = None
    route_history: list[SceneRoutePoint] = Field(default_factory=list)
    home_pad: SceneHomePad | None = None
    assets: list[SceneAsset] = Field(default_factory=list)
    airspace_zones: list[SceneAirspaceZone] = Field(default_factory=list)
    viewpoints: list[SceneViewpoint] = Field(default_factory=list)
    capture_points: list[SceneCapturePoint] = Field(default_factory=list)
    telemetry: SceneTelemetrySummary | None = None
    checklist: SceneChecklistSummary = Field(default_factory=SceneChecklistSummary)
    reward: SceneRewardSummary | None = None
    warnings: list[str] = Field(default_factory=list)


class RichSimEvent(BaseModel):
    """Small event envelope for streaming renderer updates."""

    schema_version: str = "rich_sim.event.v1"
    event_type: str
    step_count: int | None = None
    elapsed_time_s: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def build_scene_from_observation(
    observation: DroneObservation,
    route_history: RouteHistoryInput | None = None,
) -> RichSimScene:
    """Build a renderer scene from an agent-visible observation."""

    site_map = observation.site_map
    metadata = observation.metadata or {}
    return _build_scene(
        metadata=SceneMetadata(
            episode_id=_string_or_none(metadata.get("episode_id")),
            domain=_string_or_none(metadata.get("domain") or (site_map.domain if site_map else None)),
            scenario_family=_string_or_none(metadata.get("scenario_family")),
            scenario_seed=_int_or_none(metadata.get("scenario_seed")),
            step_count=_int_or_none(metadata.get("step_count")),
            done=observation.done,
        ),
        telemetry=observation.telemetry,
        mission=observation.mission,
        site_map=site_map,
        assets=observation.visible_assets,
        captures=observation.capture_log or observation.evidence_artifacts,
        checklist_status=observation.checklist_status,
        reward_breakdown=observation.reward_breakdown,
        route_history=route_history,
        warnings=observation.warnings,
        mission_phase=observation.inspection_affordances.mission_phase,
        blockers=observation.inspection_affordances.blockers,
        pending_asset_ids=observation.inspection_affordances.pending_asset_ids,
        suggested_tools=observation.inspection_affordances.suggested_tools,
        next_due_steps=observation.inspection_affordances.next_due_steps,
    )


def build_scene_from_world(
    world: EpisodeWorld,
    route_history: RouteHistoryInput | None = None,
) -> RichSimScene:
    """Build a renderer scene from an internal world using visible fields only."""

    site_map = world.visible_site_map()
    return _build_scene(
        metadata=SceneMetadata(
            episode_id=world.episode_id,
            domain=world.domain,
            scenario_family=world.scenario_family,
            scenario_seed=world.scenario_seed,
            step_count=world.step_count,
            done=world.done,
        ),
        telemetry=world.telemetry,
        mission=world.mission,
        site_map=site_map,
        assets=world.assets,
        captures=world.capture_log,
        checklist_status=world.checklist_status,
        reward_breakdown=world.reward_breakdown,
        route_history=route_history,
        warnings=_visible_world_warnings(world),
        mission_phase=None,
        blockers=[],
        pending_asset_ids=[],
        suggested_tools=[],
        next_due_steps=max(world.max_steps - world.step_count, 0),
    )


def build_scene_event(
    event_type: str,
    payload: Mapping[str, Any] | BaseModel | None = None,
    *,
    step_count: int | None = None,
    elapsed_time_s: float | None = None,
) -> RichSimEvent:
    """Build a JSON-ready event envelope for renderer timelines."""

    json_payload = to_jsonable(payload or {})
    if not isinstance(json_payload, dict):
        json_payload = {"value": json_payload}
    return RichSimEvent(
        event_type=event_type,
        step_count=step_count,
        elapsed_time_s=elapsed_time_s,
        payload=json_payload,
    )


def _build_scene(
    *,
    metadata: SceneMetadata,
    telemetry: Telemetry | None,
    mission: MissionChecklist | None,
    site_map: SiteMap | None,
    assets: Sequence[InspectableAsset],
    captures: Sequence[Capture],
    checklist_status: ChecklistStatus,
    reward_breakdown: RewardBreakdown | None,
    route_history: RouteHistoryInput | None,
    warnings: Sequence[str],
    mission_phase: str | None,
    blockers: Sequence[str],
    pending_asset_ids: Sequence[str],
    suggested_tools: Sequence[str],
    next_due_steps: int | None,
) -> RichSimScene:
    home_pose = site_map.home if site_map else None
    map_assets = list(site_map.assets if site_map else assets)
    zones = list(site_map.airspace_zones if site_map else [])
    viewpoints = list(site_map.viewpoints if site_map else [])

    return RichSimScene(
        metadata=metadata,
        drone=_scene_drone(telemetry) if telemetry is not None else None,
        route_history=_route_points(route_history),
        home_pad=SceneHomePad(pose=_scene_pose(home_pose)) if home_pose is not None else None,
        assets=[_scene_asset(asset) for asset in map_assets],
        airspace_zones=[_scene_zone(zone) for zone in zones],
        viewpoints=[_scene_viewpoint(viewpoint) for viewpoint in viewpoints],
        capture_points=[_scene_capture(capture) for capture in captures],
        telemetry=_telemetry_summary(telemetry) if telemetry is not None else None,
        checklist=_checklist_summary(
            mission=mission,
            status=checklist_status,
            mission_phase=mission_phase,
            blockers=blockers,
            pending_asset_ids=pending_asset_ids,
            suggested_tools=suggested_tools,
            next_due_steps=next_due_steps,
        ),
        reward=_reward_summary(reward_breakdown) if reward_breakdown is not None else None,
        warnings=list(warnings),
    )


def _scene_pose(pose: Pose | None) -> ScenePose:
    if pose is None:
        return ScenePose(x=0.0, y=0.0, z=0.0, yaw_deg=0.0)
    return ScenePose(x=pose.x, y=pose.y, z=pose.z, yaw_deg=pose.yaw_deg)


def _scene_gimbal(gimbal: GimbalState) -> SceneGimbal:
    return SceneGimbal(
        pitch_deg=gimbal.pitch_deg,
        yaw_deg=gimbal.yaw_deg,
        roll_deg=gimbal.roll_deg,
        frame_mode=gimbal.frame_mode,
        target_asset_id=gimbal.target_asset_id,
        roi=_scene_pose(gimbal.roi) if gimbal.roi is not None else None,
    )


def _scene_drone(telemetry: Telemetry) -> SceneDrone:
    return SceneDrone(
        pose=_scene_pose(telemetry.pose),
        gimbal=_scene_gimbal(telemetry.gimbal),
        camera_source=telemetry.camera.active_source,
        mode=telemetry.autopilot.mode,
        armed=telemetry.autopilot.armed,
        in_air=telemetry.in_air,
        landed=telemetry.landed,
    )


def _telemetry_summary(telemetry: Telemetry) -> SceneTelemetrySummary:
    return SceneTelemetrySummary(
        battery_pct=telemetry.battery.level_pct,
        voltage_v=telemetry.battery.voltage_v,
        current_a=telemetry.battery.current_a,
        consumed_mah=telemetry.battery.consumed_mah,
        reserve_pct=telemetry.battery.reserve_pct,
        ground_speed_mps=telemetry.velocity.ground_speed_mps,
        air_speed_mps=telemetry.velocity.air_speed_mps,
        elapsed_time_s=telemetry.elapsed_time_s,
        distance_flown_m=telemetry.distance_flown_m,
        weather_band=telemetry.weather_band,
        gps_fix_type=telemetry.gps.fix_type,
        satellites_visible=telemetry.gps.satellites_visible,
        link_rssi_pct=telemetry.link.rssi_pct,
        packet_loss_pct=telemetry.link.packet_loss_pct,
        command_status=telemetry.autopilot.command_status,
        last_command=telemetry.autopilot.last_command,
        storage_remaining=telemetry.camera.storage_remaining,
    )


def _scene_asset(asset: InspectableAsset) -> SceneAsset:
    return SceneAsset(
        asset_id=asset.asset_id,
        asset_type=asset.asset_type,
        label=asset.label,
        center=ScenePose(
            x=asset.geometry.center_x,
            y=asset.geometry.center_y,
            z=asset.geometry.center_z,
            yaw_deg=asset.geometry.normal_yaw_deg,
        ),
        width_m=asset.geometry.width_m,
        height_m=asset.geometry.height_m,
        normal_yaw_deg=asset.geometry.normal_yaw_deg,
        tilt_deg=asset.geometry.tilt_deg,
        required_modalities=list(asset.required_modalities),
        safe_standoff_bands=[
            SceneStandoffBand(
                name=band.name,
                min_m=band.min_m,
                max_m=band.max_m,
                preferred_m=band.preferred_m,
            )
            for band in asset.safe_standoff_bands
        ],
        visibility_tags=list(asset.visibility_tags),
        public_notes=list(asset.public_notes),
    )


def _scene_zone(zone: AirspaceZone) -> SceneAirspaceZone:
    return SceneAirspaceZone(
        zone_id=zone.zone_id,
        label=zone.label,
        zone_type=zone.zone_type,
        constraint_level=zone.constraint_level,
        reason=zone.reason,
        min_altitude_m=zone.min_altitude_m,
        max_altitude_m=zone.max_altitude_m,
        polygon_xy=[
            ScenePoint2D(x=zone.min_x, y=zone.min_y),
            ScenePoint2D(x=zone.max_x, y=zone.min_y),
            ScenePoint2D(x=zone.max_x, y=zone.max_y),
            ScenePoint2D(x=zone.min_x, y=zone.max_y),
        ],
    )


def _scene_viewpoint(viewpoint: Viewpoint) -> SceneViewpoint:
    return SceneViewpoint(
        viewpoint_id=viewpoint.viewpoint_id,
        label=viewpoint.label,
        pose=_scene_pose(viewpoint.pose),
        asset_ids=list(viewpoint.asset_ids),
        standoff_bucket=viewpoint.standoff_bucket,
        suitable_modalities=list(viewpoint.suitable_modalities),
        notes=list(viewpoint.notes),
    )


def _scene_capture(capture: Capture) -> SceneCapturePoint:
    return SceneCapturePoint(
        photo_id=capture.photo_id,
        sensor=capture.sensor,
        label=capture.label,
        pose=_scene_pose(capture.pose),
        gimbal=_scene_gimbal(capture.gimbal),
        frustum=_scene_frustum(capture),
        asset_ids=list(capture.asset_ids),
        targets_visible=list(capture.targets_visible),
        coverage_pct=capture.coverage_pct,
        quality_score=capture.quality_score,
        resolution_score=capture.resolution_score,
        view_angle_score=capture.view_angle_score,
        blur_score=capture.blur_score,
        standoff_score=capture.standoff_score,
        gsd_score=capture.gsd_score,
        glare_score=capture.glare_score,
        thermal_contrast_score=capture.thermal_contrast_score,
        detected_anomalies=list(capture.detected_anomalies),
        warnings=list(capture.warnings),
    )


def _scene_frustum(capture: Capture) -> SceneFrustum:
    if capture.sensor == "thermal":
        hfov = THERMAL_HFOV_DEG
        vfov = THERMAL_VFOV_DEG
        range_m = THERMAL_EFFECTIVE_RANGE_M
    else:
        hfov = RGB_HFOV_DEG
        vfov = RGB_VFOV_DEG
        range_m = RGB_EFFECTIVE_RANGE_M

    zoom = capture.camera.zoom_level if capture.camera is not None else 1.0
    return SceneFrustum(
        sensor=capture.sensor,
        origin=_scene_pose(capture.pose),
        yaw_deg=capture.pose.yaw_deg + capture.gimbal.yaw_deg,
        pitch_deg=capture.gimbal.pitch_deg,
        hfov_deg=hfov,
        vfov_deg=vfov,
        range_m=range_m * min(max(zoom, 1.0), 2.0),
    )


def _checklist_summary(
    *,
    mission: MissionChecklist | None,
    status: ChecklistStatus,
    mission_phase: str | None,
    blockers: Sequence[str],
    pending_asset_ids: Sequence[str],
    suggested_tools: Sequence[str],
    next_due_steps: int | None,
) -> SceneChecklistSummary:
    return SceneChecklistSummary(
        mission_id=mission.mission_id if mission is not None else None,
        instruction=mission.instruction if mission is not None else None,
        required_rows=list(mission.required_rows) if mission is not None else [],
        thermal_rows_covered=list(status.thermal_rows_covered),
        anomalies_detected=list(status.anomalies_detected),
        anomaly_rgb_pairs=dict(status.anomaly_rgb_pairs),
        anomaly_targets=dict(status.anomaly_targets),
        returned_home=status.returned_home,
        landed=status.landed,
        evidence_submitted=status.evidence_submitted,
        complete=status.complete,
        mission_phase=mission_phase,
        blockers=list(blockers),
        pending_asset_ids=list(pending_asset_ids),
        suggested_tools=list(suggested_tools),
        next_due_steps=next_due_steps,
    )


def _reward_summary(reward: RewardBreakdown) -> SceneRewardSummary:
    return SceneRewardSummary(
        total=reward.total,
        evidence_success=reward.evidence_success,
        required_coverage=reward.required_coverage,
        issue_capture=reward.issue_capture,
        operational_efficiency=reward.operational_efficiency,
        grounded_report=reward.grounded_report,
        process_reward=reward.process_reward,
        capture_quality=reward.capture_quality,
        checklist_completion=reward.checklist_completion,
        safety_compliance=reward.safety_compliance,
        battery_management=reward.battery_management,
        safety_gate=reward.safety_gate,
        integrity_gate=reward.integrity_gate,
        penalties=reward.penalties,
    )


def _route_points(route_history: RouteHistoryInput | None) -> list[SceneRoutePoint]:
    if route_history is None:
        return []
    return [_route_point(item) for item in route_history]


def _route_point(item: Pose | Mapping[str, Any] | Sequence[float]) -> SceneRoutePoint:
    if isinstance(item, Mapping):
        pose_value = item.get("pose", item)
        pose = _coerce_scene_pose(pose_value)
        elapsed = item["elapsed_time_s"] if "elapsed_time_s" in item else item.get("t")
        return SceneRoutePoint(
            pose=pose,
            elapsed_time_s=_float_or_none(elapsed),
            label=_string_or_none(item.get("label")),
            source=_string_or_none(item.get("source")),
        )
    return SceneRoutePoint(pose=_coerce_scene_pose(item))


def _coerce_scene_pose(value: Any) -> ScenePose:
    if isinstance(value, Mapping):
        return _pose_from_mapping(value)
    if all(hasattr(value, attr) for attr in ("x", "y", "z")):
        return ScenePose(
            x=float(value.x),
            y=float(value.y),
            z=float(value.z),
            yaw_deg=float(getattr(value, "yaw_deg", 0.0)),
        )
    return _pose_from_sequence(value)


def _pose_from_mapping(value: Mapping[str, Any]) -> ScenePose:
    return ScenePose(
        x=float(value.get("x", 0.0)),
        y=float(value.get("y", 0.0)),
        z=float(value.get("z", 0.0)),
        yaw_deg=float(value.get("yaw_deg", value.get("yaw", 0.0))),
    )


def _pose_from_sequence(value: Any) -> ScenePose:
    if not isinstance(value, Sequence) or len(value) < 3:
        raise TypeError("route history entries must be poses, mappings, or x/y/z sequences")
    yaw = value[3] if len(value) > 3 else 0.0
    return ScenePose(x=float(value[0]), y=float(value[1]), z=float(value[2]), yaw_deg=float(yaw))


def _visible_world_warnings(world: EpisodeWorld) -> list[str]:
    warnings: list[str] = []
    if world.telemetry.weather_band == "high":
        warnings.append("high wind band may reduce capture stability")
    if world.telemetry.battery.level_pct < world.mission.min_battery_at_done_pct + 10.0:
        warnings.append("battery reserve margin is low")
    warnings.extend(world.safety_violations[-3:])
    return warnings


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
