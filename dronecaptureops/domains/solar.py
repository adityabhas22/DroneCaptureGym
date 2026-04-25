"""Solar inspection scenario builder."""

from __future__ import annotations

import random
import uuid

from dronecaptureops.core.constants import DEFAULT_MAX_STEPS
from dronecaptureops.core.models import (
    GimbalState,
    HiddenDefect,
    MissionChecklist,
    Pose,
    RectZone,
    TargetSurface,
    Telemetry,
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
        targets = [
            TargetSurface(
                target_id=row_id,
                label=f"Solar row {row_id[-2:]}",
                center_x=30.0,
                center_y=-16.0 + idx * 8.0,
                width_m=18.0,
                height_m=4.0,
                normal_yaw_deg=180.0,
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
        restricted_zones = [
            RectZone(
                zone_id="substation_nfZ",
                label="Substation no-fly zone",
                min_x=14.0,
                min_y=-6.0,
                max_x=24.0,
                max_y=6.0,
                zone_type="no_fly",
            )
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
            targets=targets,
            restricted_zones=restricted_zones,
            hidden_defects=hidden_defects,
            weather=sample_weather(rng),
            max_steps=DEFAULT_MAX_STEPS,
        )
