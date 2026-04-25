"""High-level public tool surface."""

from __future__ import annotations

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.simulation.safety import SafetyChecker
from dronecaptureops.tools.camera_tools import CameraTools
from dronecaptureops.tools.flight_tools import FlightTools
from dronecaptureops.tools.inspection_tools import InspectionTools
from dronecaptureops.tools.registry import ToolRegistry, ToolSpec
from dronecaptureops.tools.report_tools import ReportTools


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
    registry.register(ToolSpec("estimate_view", "Estimate visible targets from current camera pose.", optional={"sensor"}, handler=camera.estimate_view))

    registry.register(ToolSpec("takeoff", "Take off to a safe altitude.", required={"altitude_m"}, handler=flight.takeoff))
    registry.register(ToolSpec("fly_to_viewpoint", "Fly to a safe local waypoint.", required={"x", "y", "z"}, optional={"yaw_deg", "speed_mps"}, handler=flight.fly_to_viewpoint))
    registry.register(ToolSpec("hover", "Hover in place.", optional={"seconds"}, handler=flight.hover))
    registry.register(ToolSpec("return_home", "Return to the home pose.", handler=flight.return_home))
    registry.register(ToolSpec("land", "Land at the current pose.", handler=flight.land))

    registry.register(ToolSpec("set_gimbal", "Set camera gimbal pitch and optional yaw.", required={"pitch_deg"}, optional={"yaw_deg"}, handler=camera.set_gimbal))
    registry.register(ToolSpec("set_zoom", "Set simulated zoom level.", required={"zoom_level"}, handler=camera.set_zoom))
    registry.register(ToolSpec("capture_rgb", "Capture RGB context or close-up evidence.", optional={"label"}, handler=camera.capture_rgb))
    registry.register(ToolSpec("capture_thermal", "Capture thermal overview evidence.", optional={"label"}, handler=camera.capture_thermal))
    registry.register(ToolSpec("inspect_capture", "Inspect structured metadata for a captured photo.", required={"photo_id"}, handler=camera.inspect_capture))

    registry.register(ToolSpec("mark_target_inspected", "Request visible validation that a target has evidence.", required={"target_id"}, optional={"photo_ids"}, handler=inspection.mark_target_inspected))
    registry.register(ToolSpec("submit_evidence_pack", "Submit final photo-linked evidence pack.", required={"summary", "photo_ids"}, optional={"findings"}, handler=report.submit_evidence_pack))
    return registry
