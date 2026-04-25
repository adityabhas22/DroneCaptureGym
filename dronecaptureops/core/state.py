"""Internal episode state."""

from __future__ import annotations

from pydantic import BaseModel, Field

from dronecaptureops.core.models import (
    AirspaceZone,
    Capture,
    ChecklistStatus,
    EvidenceReport,
    HiddenDefect,
    InspectableAsset,
    InspectionArtifact,
    MissionChecklist,
    Pose,
    RewardBreakdown,
    SiteMap,
    Telemetry,
    Viewpoint,
    WeatherState,
)


class EpisodeWorld(BaseModel):
    """Internal world state, including verifier-only fields."""

    episode_id: str
    domain: str
    scenario_family: str = "baseline_hotspot"
    scenario_seed: int
    home_pose: Pose
    telemetry: Telemetry
    mission: MissionChecklist
    assets: list[InspectableAsset]
    airspace_zones: list[AirspaceZone]
    viewpoints: list[Viewpoint] = Field(default_factory=list)
    hidden_defects: list[HiddenDefect] = Field(default_factory=list)
    true_asset_state: dict[str, dict] = Field(default_factory=dict)
    hidden_weather_details: dict[str, float | str] = Field(default_factory=dict)
    obstacle_schedule: list[dict] = Field(default_factory=list)
    verifier_evidence_requirements: list[dict] = Field(default_factory=list)
    weather: WeatherState = Field(default_factory=WeatherState)
    step_count: int = 0
    max_steps: int = 40
    elapsed_time_s: float = 0.0
    distance_flown_m: float = 0.0
    invalid_action_count: int = 0
    safety_violations: list[str] = Field(default_factory=list)
    action_log: list[dict] = Field(default_factory=list)
    observation_log: list[dict] = Field(default_factory=list)
    capture_log: list[Capture] = Field(default_factory=list)
    evidence_artifacts: list[InspectionArtifact] = Field(default_factory=list)
    checklist_status: ChecklistStatus = Field(default_factory=ChecklistStatus)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    final_report: EvidenceReport | None = None
    process_reward_total: float = 0.0
    inspected_photo_ids: list[str] = Field(default_factory=list)
    done: bool = False
    termination_reason: str | None = None

    @property
    def targets(self) -> list[InspectableAsset]:
        """Backward-compatible access to inspectable assets."""

        return self.assets

    @property
    def restricted_zones(self) -> list[AirspaceZone]:
        """Backward-compatible access to airspace zones."""

        return self.airspace_zones

    def visible_site_map(self) -> SiteMap:
        """Return only map data that the agent may observe."""

        return SiteMap(
            domain=self.domain,
            home=self.home_pose,
            assets=self.assets,
            airspace_zones=self.airspace_zones,
            viewpoints=self.viewpoints,
        )
