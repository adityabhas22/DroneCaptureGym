"""High-level tool registry exposed to agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import RawDroneAction
from dronecaptureops.core.state import EpisodeWorld


ToolHandler = Callable[[EpisodeWorld, dict[str, Any]], dict[str, Any]]


ArgType = str  # one of: "number" | "string" | "list[string]" | "boolean"


@dataclass(frozen=True)
class ArgSpec:
    """Typed schema for a single tool argument.

    Used for both LLM-readable catalog metadata and registry-level type/range
    validation. Per-handler coercion still runs as defense in depth.
    """

    type: ArgType
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ToolSpec:
    """Public tool schema."""

    name: str
    description: str
    required: set[str] = field(default_factory=set)
    optional: set[str] = field(default_factory=set)
    handler: ToolHandler | None = None
    arg_schema: dict[str, ArgSpec] = field(default_factory=dict)


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
                    "arg_schema": {
                        arg_name: {
                            "type": arg.type,
                            "description": arg.description,
                            "minimum": arg.minimum,
                            "maximum": arg.maximum,
                            "choices": list(arg.choices) if arg.choices else None,
                        }
                        for arg_name, arg in sorted(spec.arg_schema.items())
                    },
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
        for arg_name, arg in spec.arg_schema.items():
            if arg_name not in action.arguments:
                continue
            _validate_arg_value(action.tool_name, arg_name, arg, action.arguments[arg_name])
        return spec

    def execute(self, world: EpisodeWorld, action: RawDroneAction) -> dict[str, Any]:
        spec = self.validate(action)
        if spec.handler is None:
            raise ActionValidationError(f"tool has no handler: {action.tool_name}")
        return spec.handler(world, action.arguments)


def _validate_arg_value(tool_name: str, arg_name: str, arg: ArgSpec, value: Any) -> None:
    """Type/range/enum validation against an ArgSpec; raises ActionValidationError."""

    if arg.type == "number":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ActionValidationError(f"{tool_name}.{arg_name}: expected number, got {type(value).__name__}")
        numeric = float(value)
        if numeric != numeric:  # NaN
            raise ActionValidationError(f"{tool_name}.{arg_name}: NaN is not allowed")
        if arg.minimum is not None and numeric < arg.minimum:
            raise ActionValidationError(f"{tool_name}.{arg_name} below minimum {arg.minimum}: {numeric}")
        if arg.maximum is not None and numeric > arg.maximum:
            raise ActionValidationError(f"{tool_name}.{arg_name} above maximum {arg.maximum}: {numeric}")
        return
    if arg.type == "string":
        if not isinstance(value, str) or not value.strip():
            raise ActionValidationError(f"{tool_name}.{arg_name}: expected non-empty string, got {value!r}")
        if arg.choices is not None and value not in arg.choices:
            raise ActionValidationError(f"{tool_name}.{arg_name} must be one of {list(arg.choices)}, got {value!r}")
        return
    if arg.type == "list[string]":
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ActionValidationError(f"{tool_name}.{arg_name}: expected list of strings")
        return
    if arg.type == "boolean":
        if not isinstance(value, bool):
            raise ActionValidationError(f"{tool_name}.{arg_name}: expected boolean, got {type(value).__name__}")
        return
    raise ActionValidationError(f"{tool_name}.{arg_name}: unsupported schema type {arg.type!r}")
