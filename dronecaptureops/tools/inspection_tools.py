"""Mission and inspection tool handlers."""

from __future__ import annotations

from typing import Any

from dronecaptureops.core.state import EpisodeWorld


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

    def mark_target_inspected(self, world: EpisodeWorld, args: dict[str, Any]) -> dict[str, Any]:
        target_id = args["target_id"]
        real_capture_ids = {capture.photo_id for capture in world.capture_log}
        cited = [photo_id for photo_id in args.get("photo_ids", []) if photo_id in real_capture_ids]
        covered = target_id in world.checklist_status.thermal_rows_covered
        return {"target_id": target_id, "accepted": bool(covered and cited), "cited_photo_ids": cited}
