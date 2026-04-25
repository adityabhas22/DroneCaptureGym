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
