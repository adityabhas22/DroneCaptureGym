"""Lightweight episode logging helpers."""

from __future__ import annotations

from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.utils.serialization import to_jsonable


def action_to_log(action: RawDroneAction, success: bool, message: str) -> dict:
    """Build a compact action-log entry."""

    return {
        "tool_name": action.tool_name,
        "arguments": to_jsonable(action.arguments),
        "success": success,
        "message": message,
    }


def observation_to_log(observation: DroneObservation) -> dict:
    """Build a compact observation-log entry."""

    return {
        "done": observation.done,
        "reward": observation.reward,
        "system_message": observation.system_message,
        "error": observation.error,
        "checklist_status": to_jsonable(observation.checklist_status),
    }
