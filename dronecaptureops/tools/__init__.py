"""High-level public tool surface."""

from __future__ import annotations

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.simulation.safety import SafetyChecker
from dronecaptureops.tools.camera_tools import CameraTools
from dronecaptureops.tools.flight_tools import FlightTools
from dronecaptureops.tools.inspection_tools import InspectionTools
from dronecaptureops.tools.registry import ArgSpec, ToolRegistry, ToolSpec
from dronecaptureops.tools.report_tools import ReportTools


_NUM = lambda desc="", lo=None, hi=None: ArgSpec(type="number", description=desc, minimum=lo, maximum=hi)  # noqa: E731
_STR = lambda desc="", choices=None: ArgSpec(type="string", description=desc, choices=choices)  # noqa: E731
_LIST_STR = lambda desc="": ArgSpec(type="list[string]", description=desc)  # noqa: E731


def build_tool_registry(controller: DroneController, safety: SafetyChecker) -> ToolRegistry:
    """Create the initial public tool registry."""

    registry = ToolRegistry()
    flight = FlightTools(controller, safety)
    camera = CameraTools(controller, safety)
    inspection = InspectionTools()
    report = ReportTools()

    registry.register(ToolSpec("get_site_map", "Return visible site geometry.", handler=inspection.get_site_map))
    registry.register(ToolSpec("get_mission_checklist", "Return mission checklist and visible progress.", handler=inspection.get_mission_checklist))
    registry.register(ToolSpec("get_telemetry", "Return visible drone telemetry.", handler=inspection.get_telemetry))
    registry.register(ToolSpec("list_assets", "Return visible inspectable assets and pending modalities.", handler=inspection.list_assets))
    registry.register(ToolSpec(
        "estimate_view", "Estimate visible targets from current camera pose.",
        optional={"sensor"}, handler=camera.estimate_view,
        arg_schema={"sensor": _STR("Active sensor for the estimate.", choices=("rgb", "thermal", "rgb_thermal"))},
    ))
    registry.register(ToolSpec("estimate_return_margin", "Estimate distance and battery reserve needed to return home.", handler=inspection.estimate_return_margin))
    registry.register(ToolSpec(
        "request_route_replan", "Return safe route/viewpoint alternatives for a visible constraint.",
        required={"reason"}, handler=inspection.request_route_replan,
        arg_schema={"reason": _STR("Why the agent wants a replan (e.g. 'no_fly', 'obstacle').")},
    ))

    registry.register(ToolSpec(
        "takeoff", "Take off to a safe altitude.",
        required={"altitude_m"}, handler=flight.takeoff,
        arg_schema={"altitude_m": _NUM("Target altitude above home in meters.", lo=0.0, hi=120.0)},
    ))
    registry.register(ToolSpec(
        "fly_to_viewpoint", "Fly to a safe local waypoint.",
        required={"x", "y", "z"}, optional={"yaw_deg", "speed_mps"}, handler=flight.fly_to_viewpoint,
        arg_schema={
            "x": _NUM("Local east coordinate (m)."),
            "y": _NUM("Local north coordinate (m)."),
            "z": _NUM("Altitude AGL (m).", lo=0.0, hi=120.0),
            "yaw_deg": _NUM("Heading in degrees.", lo=-360.0, hi=360.0),
            "speed_mps": _NUM("Cruise speed in m/s.", lo=0.0, hi=20.0),
        },
    ))
    registry.register(ToolSpec(
        "move_to_asset", "Move to a safe standoff viewpoint for a visible asset.",
        required={"asset_id", "standoff_bucket"}, optional={"speed_mps"}, handler=flight.move_to_asset,
        arg_schema={
            "asset_id": _STR("ID of a visible InspectableAsset."),
            "standoff_bucket": _STR("One of far/mid/close.", choices=("far", "mid", "close")),
            "speed_mps": _NUM("Cruise speed in m/s.", lo=0.0, hi=20.0),
        },
    ))
    registry.register(ToolSpec(
        "hover", "Hover in place.",
        optional={"seconds"}, handler=flight.hover,
        arg_schema={"seconds": _NUM("Seconds to hover.", lo=0.0, hi=60.0)},
    ))
    registry.register(ToolSpec("return_home", "Return to the home pose.", handler=flight.return_home))
    registry.register(ToolSpec("land", "Land at the current pose.", handler=flight.land))

    registry.register(ToolSpec(
        "set_gimbal", "Set camera gimbal pitch and optional yaw.",
        required={"pitch_deg"}, optional={"yaw_deg"}, handler=camera.set_gimbal,
        arg_schema={
            "pitch_deg": _NUM("Gimbal pitch in degrees (negative = downward).", lo=-90.0, hi=30.0),
            "yaw_deg": _NUM("Gimbal yaw in degrees.", lo=-180.0, hi=180.0),
        },
    ))
    registry.register(ToolSpec(
        "set_zoom", "Set simulated zoom level.",
        required={"zoom_level"}, handler=camera.set_zoom,
        arg_schema={"zoom_level": _NUM("Zoom multiplier.", lo=1.0, hi=10.0)},
    ))
    registry.register(ToolSpec(
        "set_camera_source", "Set active camera source to rgb, thermal, or rgb_thermal.",
        required={"source"}, handler=camera.set_camera_source,
        arg_schema={"source": _STR("Active camera source.", choices=("rgb", "thermal", "rgb_thermal"))},
    ))
    registry.register(ToolSpec(
        "point_camera_at", "Point the camera/gimbal at a visible asset ROI.",
        required={"asset_id"}, handler=camera.point_camera_at,
        arg_schema={"asset_id": _STR("ID of a visible InspectableAsset.")},
    ))
    registry.register(ToolSpec(
        "capture_rgb", "Capture RGB context or close-up evidence.",
        optional={"label"}, handler=camera.capture_rgb,
        arg_schema={"label": _STR("Free-form label for the capture.")},
    ))
    registry.register(ToolSpec(
        "capture_thermal", "Capture thermal overview evidence.",
        optional={"label"}, handler=camera.capture_thermal,
        arg_schema={"label": _STR("Free-form label for the capture.")},
    ))
    registry.register(ToolSpec(
        "inspect_capture", "Inspect structured metadata for a captured photo.",
        required={"photo_id"}, handler=camera.inspect_capture,
        arg_schema={"photo_id": _STR("Photo ID returned from a previous capture.")},
    ))

    registry.register(ToolSpec(
        "mark_target_inspected", "Request visible validation that a target has evidence.",
        required={"target_id"}, optional={"photo_ids"}, handler=inspection.mark_target_inspected,
        arg_schema={
            "target_id": _STR("ID of an asset that should be acknowledged."),
            "photo_ids": _LIST_STR("Photo IDs that justify the acknowledgement."),
        },
    ))
    registry.register(
        ToolSpec(
            "submit_evidence_pack",
            "Submit final photo-linked evidence pack.",
            optional={"summary", "photo_ids", "findings", "mission_status", "evidence", "issues_found", "open_items", "safety_notes"},
            handler=report.submit_evidence_pack,
            arg_schema={
                "summary": _STR("Plain-text summary of the inspection result."),
                "photo_ids": _LIST_STR("Photo IDs cited as overall evidence."),
                "mission_status": _STR("One of complete/partial/aborted.", choices=("complete", "partial", "aborted")),
            },
        )
    )
    return registry
