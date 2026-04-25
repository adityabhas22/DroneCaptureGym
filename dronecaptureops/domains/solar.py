"""Solar inspection scenario builder."""

from __future__ import annotations

import random
import uuid

from dronecaptureops.core.constants import DEFAULT_MAX_STEPS
from dronecaptureops.core.models import (
    AirspaceZone,
    AssetGeometry,
    GimbalState,
    HiddenDefect,
    InspectableAsset,
    MissionChecklist,
    Pose,
    StandoffBand,
    Telemetry,
    Viewpoint,
)
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.domains.base import DomainScenarioBuilder
from dronecaptureops.simulation.weather import sample_weather


class SolarScenarioBuilder(DomainScenarioBuilder):
    """Build deterministic solar farm inspection scenarios."""

    domain = "solar"

    def build(self, seed: int, episode_id: str | None = None) -> EpisodeWorld:
        rng = random.Random(seed)
        row_ids = [f"row_B{idx}" for idx in range(4, 9)]
        standoff_bands = [
            StandoffBand(name="far", min_m=28.0, max_m=55.0, preferred_m=40.0),
            StandoffBand(name="mid", min_m=14.0, max_m=32.0, preferred_m=22.0),
            StandoffBand(name="close", min_m=7.0, max_m=18.0, preferred_m=12.0),
        ]
        assets = [
            InspectableAsset(
                asset_id=row_id,
                asset_type="solar_row",
                label=f"Solar row {row_id[-2:]}",
                geometry=AssetGeometry(
                    center_x=30.0,
                    center_y=-16.0 + idx * 8.0,
                    center_z=0.0,
                    width_m=18.0,
                    height_m=4.0,
                    normal_yaw_deg=180.0,
                    tilt_deg=18.0,
                ),
                required_modalities=["thermal"],
                safe_standoff_bands=standoff_bands,
                visibility_tags=["panel_row", "block_B"],
                public_notes=["Thermal overview required", "RGB close-up required if anomaly is detected"],
            )
            for idx, row_id in enumerate(row_ids)
        ]
        defect_target = rng.choice(row_ids[1:4])
        hidden_defects = [
            HiddenDefect(
                defect_id=f"hotspot_{defect_target[-2:]}",
                target_id=defect_target,
                defect_type="thermal_hotspot",
                severity=round(rng.uniform(0.65, 0.95), 2),
            )
        ]
        if rng.random() < 0.45:
            shadow_target = rng.choice([row for row in row_ids if row != defect_target])
            hidden_defects.append(
                HiddenDefect(
                    defect_id=f"shadow_{shadow_target[-2:]}",
                    target_id=shadow_target,
                    defect_type="vegetation_shadow",
                    severity=round(rng.uniform(0.35, 0.65), 2),
                )
            )

        home = Pose(x=0.0, y=0.0, z=0.0, yaw_deg=0.0)
        telemetry = Telemetry(pose=home.model_copy(deep=True), gimbal=GimbalState())
        weather = sample_weather(rng)
        telemetry.weather_band = weather.wind_band
        telemetry.sync_legacy_fields()
        airspace_zones = [
            AirspaceZone(
                zone_id="substation_nfZ",
                label="Substation no-fly zone",
                min_x=14.0,
                min_y=-6.0,
                max_x=24.0,
                max_y=6.0,
                max_altitude_m=80.0,
                zone_type="no_fly",
                constraint_level="hard",
                reason="Substation equipment clearance",
            )
        ]
        viewpoints = [
            Viewpoint(
                viewpoint_id="vp_block_b_west_overview",
                label="West overview for rows B4-B8",
                pose=Pose(x=30.0, y=24.0, z=18.0, yaw_deg=-90.0),
                asset_ids=row_ids,
                standoff_bucket="far",
                suitable_modalities=["thermal"],
                notes=["Captures all required rows when gimbal is pitched down"],
            ),
            Viewpoint(
                viewpoint_id="vp_block_b_close_context",
                label="Close RGB context for block B anomalies",
                pose=Pose(x=30.0, y=16.0, z=12.0, yaw_deg=-90.0),
                asset_ids=row_ids,
                standoff_bucket="mid",
                suitable_modalities=["rgb"],
                notes=["Use after thermal anomaly detection"],
            ),
        ]
        mission = MissionChecklist(
            mission_id="solar_block_b",
            instruction=(
                "Inspect inverter block B. Capture thermal overview of rows B4-B8. "
                "For any detected anomaly, capture RGB context or close-up evidence. "
                "Avoid the substation no-fly zone and return home with at least 20% battery."
            ),
            required_rows=row_ids,
        )
        return EpisodeWorld(
            episode_id=episode_id or str(uuid.uuid4()),
            domain=self.domain,
            scenario_seed=seed,
            home_pose=home,
            telemetry=telemetry,
            mission=mission,
            assets=assets,
            airspace_zones=airspace_zones,
            viewpoints=viewpoints,
            hidden_defects=hidden_defects,
            true_asset_state={
                asset.asset_id: {"nominal": True, "known_to_agent": False}
                for asset in assets
            },
            hidden_weather_details={"wind_mps": weather.wind_mps, "visibility": weather.visibility},
            verifier_evidence_requirements=[
                {"asset_id": row_id, "modality": "thermal", "quality_threshold": 0.55}
                for row_id in row_ids
            ],
            weather=weather,
            max_steps=DEFAULT_MAX_STEPS,
        )
