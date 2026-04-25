"""Mission and inspection tool handlers."""

from __future__ import annotations

from typing import Any

from dronecaptureops.core.coercion import coerce_str, coerce_str_list
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
        from dronecaptureops.simulation.world import active_zones  # local import to avoid cycle

        reason = coerce_str(args, "reason")
        currently_active = active_zones(world)
        hard_obstacles = [
            zone
            for zone in currently_active
            if zone.zone_type in {"no_fly", "obstacle"} and zone.constraint_level == "hard"
        ]
        blocked_zones = [zone.zone_id for zone in hard_obstacles]
        recommendations = []
        for viewpoint in world.viewpoints:
            if any(_pose_in_active_zone(viewpoint.pose, zone) for zone in hard_obstacles):
                continue
            recommendations.append(
                {
                    "viewpoint_id": viewpoint.viewpoint_id,
                    "label": viewpoint.label,
                    "asset_ids": viewpoint.asset_ids,
                    "standoff_bucket": viewpoint.standoff_bucket,
                    "suitable_modalities": viewpoint.suitable_modalities,
                    "pose": viewpoint.pose.model_dump(mode="json"),
                }
            )
        return {
            "reason": reason,
            "blocked_zone_ids": blocked_zones,
            "recommended_viewpoints": recommendations,
            "message": (
                "These viewpoints lie outside the currently active hard zones."
                if recommendations
                else "No viewpoints currently lie outside the active hard zones; consider waiting for an obstacle window to clear."
            ),
        }

    def mark_target_inspected(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        """Acknowledge inspection of a target with explicit photo evidence.

        The agent supplies a target_id and a list of photo_ids that justify
        the inspection. We accept the acknowledgement only when the cited
        photos exist, the target is visible in at least one of them, and
        the per-target quality clears MIN_ROW_QUALITY for at least one
        thermal photo.
        """

        from dronecaptureops.rewards.verifiers import MIN_ROW_QUALITY  # local import to avoid cycle

        target_id = coerce_str(args, "target_id")
        cited_ids = coerce_str_list(args, "photo_ids")
        real_captures = {capture.photo_id: capture for capture in world.capture_log}
        unknown = [photo_id for photo_id in cited_ids if photo_id not in real_captures]
        cited = [photo_id for photo_id in cited_ids if photo_id in real_captures]
        target_known = any(asset.asset_id == target_id for asset in world.assets)
        thermal_evidence = [
            real_captures[photo_id]
            for photo_id in cited
            if real_captures[photo_id].sensor == "thermal"
            and target_id in real_captures[photo_id].targets_visible
            and real_captures[photo_id].target_quality(target_id) >= MIN_ROW_QUALITY
        ]
        accepted = bool(target_known and thermal_evidence)
        if accepted and target_id not in world.checklist_status.targets_acknowledged:
            world.checklist_status.targets_acknowledged.append(target_id)
            world.checklist_status.targets_acknowledged.sort()
        warnings: list[str] = []
        if not target_known:
            warnings.append(f"unknown target_id: {target_id}")
        if unknown:
            warnings.append(f"unknown photo_ids: {sorted(unknown)}")
        if target_known and not thermal_evidence:
            warnings.append("no cited thermal photo proves this target was inspected")
        return {
            "target_id": target_id,
            "accepted": accepted,
            "cited_photo_ids": cited,
            "warnings": warnings,
        }


def _pose_in_active_zone(pose, zone) -> bool:
    return (
        zone.min_x <= pose.x <= zone.max_x
        and zone.min_y <= pose.y <= zone.max_y
        and zone.min_altitude_m <= pose.z <= zone.max_altitude_m
    )
