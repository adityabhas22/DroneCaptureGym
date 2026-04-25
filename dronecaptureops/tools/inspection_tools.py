"""Mission and inspection tool handlers."""

from __future__ import annotations

from typing import Any

from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.utils.math_utils import distance_3d


class InspectionTools:
    """Tools for visible mission/map state."""

    def get_site_map(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        return world.visible_site_map().model_dump(mode="json")

    def get_mission_checklist(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "mission": world.mission.model_dump(mode="json"),
            "status": world.checklist_status.model_dump(mode="json"),
        }

    def get_telemetry(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        return world.telemetry.model_dump(mode="json")

    def list_assets(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        covered = set(world.checklist_status.thermal_rows_covered)
        return {
            "assets": [
                {
                    "asset_id": asset.asset_id,
                    "asset_type": asset.asset_type,
                    "label": asset.label,
                    "required_modalities": asset.required_modalities,
                    "pending_modalities": [] if asset.asset_id in covered else asset.required_modalities,
                    "safe_standoff_bands": [band.model_dump(mode="json") for band in asset.safe_standoff_bands],
                    "visibility_tags": asset.visibility_tags,
                    "public_notes": asset.public_notes,
                }
                for asset in world.assets
            ],
            "pending_asset_ids": [
                asset.asset_id for asset in world.assets if asset.asset_id not in covered
            ],
        }

    def estimate_return_margin(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        current = world.telemetry.pose
        home = world.home_pose.model_copy(deep=True)
        home.z = max(current.z, 10.0)
        distance_m = distance_3d(current, home)
        estimated_battery_needed_pct = round(distance_m * 0.045 + 0.4, 3)
        reserve_after_return_pct = round(world.telemetry.battery.level_pct - estimated_battery_needed_pct, 3)
        return {
            "distance_home_m": round(distance_m, 3),
            "estimated_battery_needed_pct": estimated_battery_needed_pct,
            "reserve_after_return_pct": reserve_after_return_pct,
            "meets_required_reserve": reserve_after_return_pct >= world.mission.min_battery_at_done_pct,
        }

    def request_route_replan(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        reason = args["reason"]
        blocked_zones = [
            zone.zone_id
            for zone in world.airspace_zones
            if zone.zone_type in {"no_fly", "obstacle"} and zone.constraint_level == "hard"
        ]
        recommendations = [
            {
                "viewpoint_id": viewpoint.viewpoint_id,
                "label": viewpoint.label,
                "asset_ids": viewpoint.asset_ids,
                "standoff_bucket": viewpoint.standoff_bucket,
                "suitable_modalities": viewpoint.suitable_modalities,
            }
            for viewpoint in world.viewpoints
        ]
        return {
            "reason": reason,
            "blocked_zone_ids": blocked_zones,
            "recommended_viewpoints": recommendations,
            "message": "Use named viewpoints that avoid hard no-fly or obstacle zones.",
        }

    def mark_target_inspected(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        target_id = args["target_id"]
        real_capture_ids = {capture.photo_id for capture in world.capture_log}
        cited = [photo_id for photo_id in args.get("photo_ids", []) if photo_id in real_capture_ids]
        covered = target_id in world.checklist_status.thermal_rows_covered
        return {"target_id": target_id, "accepted": bool(covered and cited), "cited_photo_ids": cited}
