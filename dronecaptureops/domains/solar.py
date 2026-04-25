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
    WeatherState,
)
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.domains.base import DomainScenarioBuilder
from dronecaptureops.generation.suites import SOLAR_SCENARIO_FAMILIES
from dronecaptureops.simulation.weather import sample_weather


class SolarScenarioBuilder(DomainScenarioBuilder):
    """Build deterministic solar farm inspection scenarios."""

    domain = "solar"

    def build(self, seed: int, episode_id: str | None = None, scenario_family: str | None = None) -> EpisodeWorld:
        rng = random.Random(seed)
        family = scenario_family or _family_for_seed(seed)
        difficulty = _difficulty_for_seed(seed)
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
                required_modalities=["thermal", "rgb"] if family in {"bypass_diode_fault", "false_positive_glare"} else ["thermal"],
                safe_standoff_bands=standoff_bands,
                visibility_tags=["panel_row", "block_B", family],
                public_notes=["Thermal overview required", "RGB close-up required if anomaly is detected"],
            )
            for idx, row_id in enumerate(row_ids)
        ]
        hidden_defects = _hidden_defects_for_family(family, row_ids, rng)

        home = Pose(x=0.0, y=0.0, z=0.0, yaw_deg=0.0)
        telemetry = Telemetry(pose=home.model_copy(deep=True), gimbal=GimbalState())
        weather = _weather_for_family(family, difficulty, sample_weather(rng))
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
        if family == "blocked_corridor_replan":
            airspace_zones.append(
                AirspaceZone(
                    zone_id="temporary_crane_corridor",
                    label="Temporary crane corridor",
                    min_x=25.0,
                    min_y=4.0,
                    max_x=36.0,
                    max_y=20.0,
                    max_altitude_m=35.0,
                    zone_type="obstacle",
                    constraint_level="hard",
                    reason="Maintenance crane blocks the direct close-up route.",
                )
            )
        if family == "false_positive_glare":
            airspace_zones.append(
                AirspaceZone(
                    zone_id="neighbor_property_privacy",
                    label="Neighbor privacy buffer",
                    min_x=38.0,
                    min_y=-28.0,
                    max_x=48.0,
                    max_y=28.0,
                    max_altitude_m=80.0,
                    zone_type="privacy",
                    constraint_level="soft",
                    reason="Avoid unnecessary RGB captures beyond the solar boundary.",
                )
            )
        # Multi-viewpoint thermal scan: a real thermal camera has ~30° vertical
        # FOV, so a single capture cannot frame all five rows. Two viewpoints
        # NORTH and SOUTH of the row block cover three rows each from a steep
        # gimbal pitch (~-56°), with row B6 in the overlap. RGB close-ups
        # provide anomaly context for either half.
        viewpoints = [
            Viewpoint(
                viewpoint_id="vp_block_b_north_overview",
                label="North overview covering rows B6-B8",
                pose=Pose(x=30.0, y=24.0, z=22.0, yaw_deg=-90.0),
                asset_ids=["row_B6", "row_B7", "row_B8"],
                standoff_bucket="far",
                suitable_modalities=["thermal"],
                notes=[
                    "Pitch the gimbal to about -56 degrees to frame rows B6-B8.",
                    "Pair with vp_block_b_south_overview for full row coverage.",
                ],
            ),
            Viewpoint(
                viewpoint_id="vp_block_b_south_overview",
                label="South overview covering rows B4-B6",
                pose=Pose(x=30.0, y=-24.0, z=22.0, yaw_deg=90.0),
                asset_ids=["row_B4", "row_B5", "row_B6"],
                standoff_bucket="far",
                suitable_modalities=["thermal"],
                notes=[
                    "Pitch the gimbal to about -56 degrees to frame rows B4-B6.",
                    "Pair with vp_block_b_north_overview for full row coverage.",
                ],
            ),
            Viewpoint(
                viewpoint_id="vp_block_b_close_north",
                label="Close-up RGB context for rows B6-B8",
                pose=Pose(x=30.0, y=24.0, z=14.0, yaw_deg=-90.0),
                asset_ids=["row_B6", "row_B7", "row_B8"],
                standoff_bucket="mid",
                suitable_modalities=["rgb"],
                notes=["Pitch the gimbal to about -45 degrees for RGB close-up of rows B6-B8."],
            ),
            Viewpoint(
                viewpoint_id="vp_block_b_close_south",
                label="Close-up RGB context for rows B4-B6",
                pose=Pose(x=30.0, y=-24.0, z=14.0, yaw_deg=90.0),
                asset_ids=["row_B4", "row_B5", "row_B6"],
                standoff_bucket="mid",
                suitable_modalities=["rgb"],
                notes=["Pitch the gimbal to about -45 degrees for RGB close-up of rows B4-B6."],
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
            scenario_family=family,
            difficulty=difficulty,
            environmental_constraints=_environmental_constraints(family, weather),
        )
        if family == "low_battery_tradeoff":
            telemetry.battery.level_pct = 68.0
            telemetry.battery_pct = 68.0
        return EpisodeWorld(
            episode_id=episode_id or str(uuid.uuid4()),
            domain=self.domain,
            scenario_family=family,
            scenario_seed=seed,
            home_pose=home,
            telemetry=telemetry,
            mission=mission,
            assets=assets,
            airspace_zones=airspace_zones,
            viewpoints=viewpoints,
            hidden_defects=hidden_defects,
            true_asset_state={
                asset.asset_id: {
                    "nominal": True,
                    "known_to_agent": False,
                    "scenario_family": family,
                }
                for asset in assets
            },
            hidden_weather_details={
                "wind_mps": weather.wind_mps,
                "visibility": weather.visibility,
                "irradiance_wm2": weather.irradiance_wm2,
                "cloud_cover_oktas": weather.cloud_cover_oktas,
            },
            obstacle_schedule=_obstacle_schedule(family),
            verifier_evidence_requirements=[
                {"asset_id": row_id, "modality": "thermal", "quality_threshold": 0.55}
                for row_id in row_ids
            ],
            weather=weather,
            max_steps=DEFAULT_MAX_STEPS,
        )


def _family_for_seed(seed: int) -> str:
    if seed in {2101, 2102, 2103}:
        return ("single_hotspot", "soiling_and_shadow", "bypass_diode_fault")[seed - 2101]
    if seed in {2201, 2202, 2203}:
        return ("soiling_and_shadow", "bypass_diode_fault", "false_positive_glare")[seed - 2201]
    if seed in {2301, 2302, 2303}:
        return ("blocked_corridor_replan", "low_battery_tradeoff", "false_positive_glare")[seed - 2301]
    return SOLAR_SCENARIO_FAMILIES[seed % len(SOLAR_SCENARIO_FAMILIES)]


def _difficulty_for_seed(seed: int) -> str:
    if 2300 <= seed < 2400:
        return "hard"
    if 2200 <= seed < 2300:
        return "medium"
    return "easy"


def _hidden_defects_for_family(family: str, row_ids: list[str], rng: random.Random) -> list[HiddenDefect]:
    primary = rng.choice(row_ids[1:4])
    defects: list[HiddenDefect] = []
    if family in {"single_hotspot", "blocked_corridor_replan", "low_battery_tradeoff"}:
        defects.append(HiddenDefect(defect_id=f"hotspot_{primary[-2:]}", target_id=primary, defect_type="thermal_hotspot", severity=round(rng.uniform(0.7, 0.95), 2)))
    elif family == "soiling_and_shadow":
        defects.append(HiddenDefect(defect_id=f"soiling_{primary[-2:]}", target_id=primary, defect_type="soiling_heating", severity=round(rng.uniform(0.45, 0.7), 2)))
        shadow = rng.choice([row for row in row_ids if row != primary])
        defects.append(HiddenDefect(defect_id=f"shadow_{shadow[-2:]}", target_id=shadow, defect_type="vegetation_shadow", severity=round(rng.uniform(0.35, 0.65), 2)))
    elif family == "bypass_diode_fault":
        # Diode faults need close-standoff thermal evidence; the camera sim
        # also gates this defect type on distance and contrast.
        defects.append(
            HiddenDefect(
                defect_id=f"diode_{primary[-2:]}",
                target_id=primary,
                defect_type="bypass_diode_fault",
                severity=round(rng.uniform(0.7, 0.9), 2),
                min_quality=0.65,
            )
        )
    elif family == "false_positive_glare":
        defects.append(
            HiddenDefect(
                defect_id=f"glare_artifact_{primary[-2:]}",
                target_id=primary,
                defect_type="false_thermal_artifact",
                severity=round(rng.uniform(0.25, 0.45), 2),
                requires_rgb_context=False,
                weight=0.0,
                counts_for_issue_reward=False,
            )
        )
    return defects


def _weather_for_family(family: str, difficulty: str, sampled: WeatherState) -> WeatherState:
    if family == "low_battery_tradeoff":
        return WeatherState(wind_mps=5.8, visibility=0.86, irradiance_wm2=650.0, cloud_cover_oktas=2, ambient_temp_c=34.0)
    if family == "false_positive_glare":
        return WeatherState(wind_mps=2.4, visibility=0.96, irradiance_wm2=920.0, cloud_cover_oktas=0, ambient_temp_c=32.0)
    if difficulty == "hard":
        return WeatherState(wind_mps=6.2, visibility=0.82, irradiance_wm2=630.0, cloud_cover_oktas=2, ambient_temp_c=35.0)
    if difficulty == "medium":
        return WeatherState(wind_mps=4.0, visibility=0.9, irradiance_wm2=720.0, cloud_cover_oktas=1, ambient_temp_c=31.0)
    sampled.irradiance_wm2 = 800.0
    sampled.cloud_cover_oktas = 1
    return sampled


def _environmental_constraints(family: str, weather: WeatherState) -> list[str]:
    constraints = [
        f"wind_band:{weather.wind_band}",
        f"irradiance_wm2:{int(weather.irradiance_wm2)}",
        f"cloud_cover_oktas:{weather.cloud_cover_oktas}",
    ]
    if family == "false_positive_glare":
        constraints.append("glare_risk:high")
    if family == "blocked_corridor_replan":
        constraints.append("temporary_obstacle:crane_corridor")
    if family == "low_battery_tradeoff":
        constraints.append("battery_margin:tight")
    return constraints


def _obstacle_schedule(family: str) -> list[dict]:
    if family != "blocked_corridor_replan":
        return []
    return [
        {
            "event_id": "crane_active_window",
            "zone_id": "temporary_crane_corridor",
            "active_from_step": 0,
            "active_until_step": 30,
        }
    ]
