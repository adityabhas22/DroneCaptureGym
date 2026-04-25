"""OpenAI / Anthropic tool schemas built from the live ToolRegistry.

Single source of truth: `ToolRegistry` declares each tool's required/optional
args; this module attaches per-arg type metadata and emits JSON Schemas
suitable for OpenAI function-calling and Anthropic tool_use. If a tool is
added to `tools/__init__.py` and not annotated here, schema generation
raises so the omission surfaces in tests.
"""

from __future__ import annotations

from typing import Any

from dronecaptureops.tools.registry import ToolRegistry


# Per-tool parameter metadata. Types map to JSON Schema directly. Enums and
# numeric ranges are surfaced when the tool actually constrains them so the
# model's grammar-aware decoders can sample valid values.
_PARAM_METADATA: dict[str, dict[str, dict[str, Any]]] = {
    "get_site_map": {},
    "get_mission_checklist": {},
    "get_telemetry": {},
    "list_assets": {},
    "estimate_view": {
        "sensor": {"type": "string", "enum": ["rgb", "thermal"], "description": "Which sensor's view to estimate."}
    },
    "estimate_return_margin": {},
    "request_route_replan": {
        "reason": {"type": "string", "description": "Why a replan is requested (e.g. 'crane_corridor blocks west route')."}
    },
    "takeoff": {
        "altitude_m": {"type": "number", "minimum": 2.0, "maximum": 60.0, "description": "Target takeoff altitude in metres."}
    },
    "fly_to_viewpoint": {
        "x": {"type": "number", "description": "Local-frame x coordinate (metres east of home)."},
        "y": {"type": "number", "description": "Local-frame y coordinate (metres north of home)."},
        "z": {"type": "number", "minimum": 2.0, "maximum": 60.0, "description": "Altitude above ground in metres."},
        "yaw_deg": {"type": "number", "minimum": -180.0, "maximum": 180.0, "description": "Body yaw in degrees (0=east, 90=north, -90=south, 180=west)."},
        "speed_mps": {"type": "number", "minimum": 0.5, "maximum": 12.0, "description": "Cruise speed in metres per second."},
    },
    "move_to_asset": {
        "asset_id": {"type": "string", "description": "Asset to approach (e.g. 'row_B6')."},
        "standoff_bucket": {"type": "string", "enum": ["far", "mid", "close"], "description": "Standoff band to use for the approach viewpoint."},
        "speed_mps": {"type": "number", "minimum": 0.5, "maximum": 12.0, "description": "Cruise speed in metres per second."},
    },
    "hover": {
        "seconds": {"type": "number", "minimum": 0.0, "maximum": 60.0, "description": "Seconds to hover."}
    },
    "return_home": {},
    "land": {},
    "set_gimbal": {
        "pitch_deg": {"type": "number", "minimum": -90.0, "maximum": 20.0, "description": "Gimbal pitch in degrees (negative = looking down)."},
        "yaw_deg": {"type": "number", "minimum": -180.0, "maximum": 180.0, "description": "Gimbal yaw relative to body frame in degrees."},
    },
    "set_zoom": {
        "zoom_level": {"type": "number", "minimum": 1.0, "maximum": 4.0, "description": "Optical zoom multiplier (1.0 = wide, 4.0 = tightest)."}
    },
    "set_camera_source": {
        "source": {"type": "string", "enum": ["rgb", "thermal", "rgb_thermal"], "description": "Active sensor source."}
    },
    "point_camera_at": {
        "asset_id": {"type": "string", "description": "Asset to centre in the camera frame."}
    },
    "capture_rgb": {
        "label": {"type": "string", "description": "Free-form note attached to the capture (e.g. 'rgb close-up north')."}
    },
    "capture_thermal": {
        "label": {"type": "string", "description": "Free-form note attached to the capture."}
    },
    "inspect_capture": {
        "photo_id": {"type": "string", "description": "Photo ID returned by a previous capture call (e.g. 'IMG-T-001')."}
    },
    "mark_target_inspected": {
        "target_id": {"type": "string", "description": "Asset to mark inspected (e.g. 'row_B6')."},
        "photo_ids": {"type": "array", "items": {"type": "string"}, "description": "Photo IDs that justify the inspection."},
    },
    "submit_evidence_pack": {
        "summary": {"type": "string", "description": "Free-form summary of the inspection outcome."},
        "photo_ids": {"type": "array", "items": {"type": "string"}, "description": "All cited photo IDs."},
        "findings": {"type": "array", "items": {"type": "object"}, "description": "List of findings (compatible payload). Each item should reference a defect_id and photo_ids."},
        "mission_status": {"type": "string", "description": "Optional structured status field ('complete', 'partial', etc.)."},
        "evidence": {"type": "array", "items": {"type": "object"}, "description": "Structured requirement evidence list (each item has requirement_id, status, photo_ids)."},
        "issues_found": {"type": "array", "items": {"type": "object"}, "description": "Structured issues list (each item has issue_id, evidence_photo_ids, recommended_followup)."},
        "open_items": {"type": "array", "items": {"type": "object"}, "description": "Open items to flag for follow-up."},
        "safety_notes": {"type": "array", "items": {"type": "string"}, "description": "Procedural notes (returned home, battery reserve, etc.)."},
    },
}


def openai_tool_schemas(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Build OpenAI function-calling tool schemas for every registered tool."""

    schemas: list[dict[str, Any]] = []
    for name in registry.names():
        spec = registry.spec(name)
        params = _build_parameters(name, spec.required, spec.optional)
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec.description,
                    "parameters": params,
                },
            }
        )
    return schemas


def anthropic_tool_schemas(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Build Anthropic tool_use schemas (input_schema instead of parameters)."""

    schemas: list[dict[str, Any]] = []
    for name in registry.names():
        spec = registry.spec(name)
        params = _build_parameters(name, spec.required, spec.optional)
        schemas.append(
            {
                "name": name,
                "description": spec.description,
                "input_schema": params,
            }
        )
    return schemas


def _build_parameters(tool_name: str, required: set[str], optional: set[str]) -> dict[str, Any]:
    """Assemble a JSON Schema object for one tool's parameters."""

    if tool_name not in _PARAM_METADATA:
        raise KeyError(
            f"tool {tool_name!r} has no parameter metadata in agent.schemas; "
            "add an entry to _PARAM_METADATA"
        )
    metadata = _PARAM_METADATA[tool_name]
    expected_args = required | optional
    declared_args = set(metadata)
    missing = expected_args - declared_args
    extra = declared_args - expected_args
    if missing:
        raise KeyError(f"tool {tool_name!r} is missing parameter metadata for: {sorted(missing)}")
    if extra:
        raise KeyError(
            f"tool {tool_name!r} has stale parameter metadata for: {sorted(extra)} (not in registry)"
        )
    properties = {arg: metadata[arg] for arg in sorted(expected_args)}
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = sorted(required)
    return schema
