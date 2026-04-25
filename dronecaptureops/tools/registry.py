"""High-level tool registry exposed to agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import RawDroneAction
from dronecaptureops.core.state import EpisodeWorld


ToolHandler = Callable[[EpisodeWorld, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class ToolSpec:
    """Public tool schema."""

    name: str
    description: str
    required: set[str] = field(default_factory=set)
    optional: set[str] = field(default_factory=set)
    handler: ToolHandler | None = None


class ToolRegistry:
    """Registry for validating and executing public tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog_as_json(self, world: EpisodeWorld | None = None) -> list[dict[str, Any]]:
        """Return an LLM-friendly tool catalog."""

        availability = self.action_availability(world) if world is not None else {}
        catalog = []
        for name in self.names():
            spec = self._tools[name]
            catalog.append(
                {
                    "name": name,
                    "description": spec.description,
                    "required_args": sorted(spec.required),
                    "optional_args": sorted(spec.optional),
                    "available": availability.get(name, True),
                }
            )
        return catalog

    def action_availability(self, world: EpisodeWorld | None) -> dict[str, bool]:
        """Compute visible action availability from public state."""

        availability = {name: True for name in self._tools}
        if world is None:
            return availability
        if world.done:
            return {name: False for name in self._tools}
        in_air = world.telemetry.in_air
        landed = world.telemetry.landed
        has_capture = bool(world.capture_log)
        for name in ["fly_to_viewpoint", "move_to_asset", "hover", "return_home", "capture_rgb", "capture_thermal"]:
            if name in availability:
                availability[name] = in_air
        for name in ["takeoff"]:
            if name in availability:
                availability[name] = landed and not in_air
        if "land" in availability:
            availability["land"] = in_air
        if "inspect_capture" in availability:
            availability["inspect_capture"] = has_capture
        if "submit_evidence_pack" in availability:
            availability["submit_evidence_pack"] = bool(world.capture_log) and world.checklist_status.returned_home and world.checklist_status.landed
        return availability

    def spec(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise ActionValidationError(f"unknown tool: {name}")
        return self._tools[name]

    def validate(self, action: RawDroneAction) -> ToolSpec:
        spec = self.spec(action.tool_name)
        arguments = set(action.arguments)
        missing = spec.required - arguments
        allowed = spec.required | spec.optional
        unknown = arguments - allowed
        if missing:
            raise ActionValidationError(f"{action.tool_name} missing required arguments: {sorted(missing)}")
        if unknown:
            raise ActionValidationError(f"{action.tool_name} received unknown arguments: {sorted(unknown)}")
        return spec

    def execute(self, world: EpisodeWorld, action: RawDroneAction) -> dict[str, Any]:
        spec = self.validate(action)
        if spec.handler is None:
            raise ActionValidationError(f"tool has no handler: {action.tool_name}")
        return spec.handler(world, action.arguments)
