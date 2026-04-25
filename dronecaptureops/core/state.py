"""Internal episode state."""

from __future__ import annotations

from pydantic import BaseModel, Field

from dronecaptureops.core.models import (
    Capture,
    ChecklistStatus,
    EvidenceReport,
    HiddenDefect,
    MissionChecklist,
    Pose,
    RectZone,
    RewardBreakdown,
    SiteMap,
    TargetSurface,
    Telemetry,
    WeatherState,
)


class EpisodeWorld(BaseModel):
    """Internal world state, including verifier-only fields."""

    episode_id: str
    domain: str
    scenario_seed: int
    home_pose: Pose
    telemetry: Telemetry
    mission: MissionChecklist
    targets: list[TargetSurface]
    restricted_zones: list[RectZone]
    hidden_defects: list[HiddenDefect] = Field(default_factory=list)
    weather: WeatherState = Field(default_factory=WeatherState)
    step_count: int = 0
    max_steps: int = 40
    distance_flown_m: float = 0.0
    invalid_action_count: int = 0
    safety_violations: list[str] = Field(default_factory=list)
    action_log: list[dict] = Field(default_factory=list)
    observation_log: list[dict] = Field(default_factory=list)
    capture_log: list[Capture] = Field(default_factory=list)
    checklist_status: ChecklistStatus = Field(default_factory=ChecklistStatus)
    reward_breakdown: RewardBreakdown = Field(default_factory=RewardBreakdown)
    final_report: EvidenceReport | None = None
    done: bool = False
    termination_reason: str | None = None

    def visible_site_map(self) -> SiteMap:
        """Return only map data that the agent may observe."""

        return SiteMap(
            domain=self.domain,
            home=self.home_pose,
            targets=self.targets,
            restricted_zones=self.restricted_zones,
        )
