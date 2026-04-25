"""Task-conditioned solar inspection missions."""

from __future__ import annotations

from dataclasses import dataclass, field

from dronecaptureops.core.constants import DEFAULT_MAX_STEPS, DEFAULT_TASK


@dataclass(frozen=True)
class DefectSpec:
    """Deterministic hidden defect for a solar task."""

    defect_id: str
    target_id: str
    defect_type: str
    severity: float


@dataclass(frozen=True)
class ZoneSpec:
    """Additional visible airspace zone for a task."""

    zone_id: str
    label: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_altitude_m: float = 0.0
    max_altitude_m: float = 80.0
    zone_type: str = "obstacle"
    constraint_level: str = "hard"
    reason: str = ""


@dataclass(frozen=True)
class ViewpointSpec:
    """Additional named capture viewpoint for a task."""

    viewpoint_id: str
    label: str
    x: float
    y: float
    z: float
    yaw_deg: float
    asset_ids: tuple[str, ...]
    standoff_bucket: str = "mid"
    suitable_modalities: tuple[str, ...] = ("rgb", "thermal")
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SolarTaskSpec:
    """Visible mission and deterministic scenario modifications."""

    task_id: str
    name: str
    instruction: str
    success_criteria: tuple[str, ...]
    public_constraints: tuple[str, ...]
    task_tags: tuple[str, ...] = ()
    required_rows: tuple[str, ...] = ("row_B4", "row_B5", "row_B6", "row_B7", "row_B8")
    hidden_defects: tuple[DefectSpec, ...] | None = None
    weather_wind_mps: float | None = None
    weather_visibility: float | None = None
    initial_battery_pct: float = 100.0
    min_battery_at_done_pct: float = 20.0
    max_steps: int = DEFAULT_MAX_STEPS
    min_capture_quality: float = 0.55
    min_rgb_quality: float = 0.55
    min_report_grounding_score: float = 0.6
    thermal_overview_required: bool = True
    rgb_closeup_for_anomalies: bool = True
    must_return_home: bool = True
    extra_zones: tuple[ZoneSpec, ...] = ()
    extra_viewpoints: tuple[ViewpointSpec, ...] = ()
    verifier_notes: tuple[str, ...] = field(default_factory=tuple)


BASE_SUCCESS = (
    "Cover every required row with accepted thermal evidence.",
    "Cite only real captured photo IDs in the evidence pack.",
    "Avoid hard safety violations.",
    "Return home and land before final submission.",
)

BASE_CONSTRAINTS = (
    "The substation no-fly zone is a hard flight constraint.",
    "Hidden defects are only discoverable through valid captures.",
)

EAST_RGB_VIEWPOINTS = (
    ViewpointSpec(
        viewpoint_id="vp_row_b4_east_rgb",
        label="East RGB close-up for row B4",
        x=45.0,
        y=-16.0,
        z=16.0,
        yaw_deg=180.0,
        asset_ids=("row_B4",),
        standoff_bucket="close",
        suitable_modalities=("rgb",),
        notes=("Use for edge-row confirmation.",),
    ),
    ViewpointSpec(
        viewpoint_id="vp_row_b5_east_rgb",
        label="East RGB close-up for row B5",
        x=45.0,
        y=-8.0,
        z=16.0,
        yaw_deg=180.0,
        asset_ids=("row_B5",),
        standoff_bucket="close",
        suitable_modalities=("rgb",),
        notes=("Use for target-specific anomaly confirmation.",),
    ),
    ViewpointSpec(
        viewpoint_id="vp_row_b6_east_rgb",
        label="East RGB close-up for row B6",
        x=45.0,
        y=0.0,
        z=16.0,
        yaw_deg=180.0,
        asset_ids=("row_B6",),
        standoff_bucket="close",
        suitable_modalities=("rgb",),
        notes=("Use for target-specific anomaly confirmation.",),
    ),
    ViewpointSpec(
        viewpoint_id="vp_row_b7_east_rgb",
        label="East RGB close-up for row B7",
        x=45.0,
        y=8.0,
        z=16.0,
        yaw_deg=180.0,
        asset_ids=("row_B7",),
        standoff_bucket="close",
        suitable_modalities=("rgb",),
        notes=("Use for target-specific anomaly confirmation.",),
    ),
    ViewpointSpec(
        viewpoint_id="vp_row_b8_east_rgb",
        label="East RGB close-up for row B8",
        x=45.0,
        y=16.0,
        z=16.0,
        yaw_deg=180.0,
        asset_ids=("row_B8",),
        standoff_bucket="close",
        suitable_modalities=("rgb",),
        notes=("Use for edge-row confirmation.",),
    ),
)


SOLAR_TASKS: dict[str, SolarTaskSpec] = {
    "basic_thermal_survey": SolarTaskSpec(
        task_id="basic_thermal_survey",
        name="Basic Thermal Survey",
        instruction=(
            "Inspect inverter block B. Capture thermal overview evidence for rows B4-B8, "
            "avoid the substation no-fly zone, return home with reserve battery, land, "
            "and submit a photo-linked evidence pack."
        ),
        success_criteria=BASE_SUCCESS,
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("coverage", "baseline", "thermal"),
        hidden_defects=None,
    ),
    "anomaly_confirmation": SolarTaskSpec(
        task_id="anomaly_confirmation",
        name="Anomaly Confirmation Mission",
        instruction=(
            "Survey rows B4-B8 with thermal imagery. If a thermal anomaly is detected, "
            "collect target-specific RGB context for the affected row before submitting."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect the hidden thermal hotspot.",
            "Pair every detected anomaly with RGB evidence showing the same row.",
        ),
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("anomaly", "rgb-confirmation"),
        hidden_defects=(DefectSpec("hotspot_B6", "row_B6", "thermal_hotspot", 0.91),),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "low_battery_inspection": SolarTaskSpec(
        task_id="low_battery_inspection",
        name="Low Battery Inspection",
        instruction=(
            "Complete a thermal survey of rows B4-B8 with limited battery. Use an efficient "
            "route, avoid redundant captures, return home above the stricter reserve, and land."
        ),
        success_criteria=BASE_SUCCESS
        + ("Finish with at least 35 percent battery.", "Avoid unnecessary repositioning."),
        public_constraints=BASE_CONSTRAINTS + ("Starting battery is limited.",),
        task_tags=("battery", "efficiency"),
        hidden_defects=(),
        initial_battery_pct=45.0,
        min_battery_at_done_pct=35.0,
        max_steps=28,
        rgb_closeup_for_anomalies=False,
    ),
    "bad_weather_recapture": SolarTaskSpec(
        task_id="bad_weather_recapture",
        name="Bad Weather Re-Capture",
        instruction=(
            "Inspect rows B4-B8 during poor visibility and high wind. Review capture quality "
            "and recapture if evidence is missing, blurry, or too weak for the checklist."
        ),
        success_criteria=BASE_SUCCESS
        + ("Recover from any low-quality or incomplete capture before submission.",),
        public_constraints=BASE_CONSTRAINTS + ("High wind may reduce capture stability.",),
        task_tags=("weather", "recovery", "quality"),
        hidden_defects=(DefectSpec("hotspot_B6", "row_B6", "thermal_hotspot", 0.88),),
        weather_wind_mps=6.4,
        weather_visibility=0.82,
        min_capture_quality=0.62,
        min_rgb_quality=0.60,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "safety_constrained_route": SolarTaskSpec(
        task_id="safety_constrained_route",
        name="Safety-Constrained Route Planning",
        instruction=(
            "Plan a safe route around the substation no-fly zone before collecting thermal "
            "evidence for rows B4-B8. Direct unsafe shortcuts should be avoided."
        ),
        success_criteria=BASE_SUCCESS + ("Use a safe corridor instead of crossing a hard zone.",),
        public_constraints=BASE_CONSTRAINTS
        + ("The direct path from home to the target block crosses restricted airspace.",),
        task_tags=("safety", "routing"),
        hidden_defects=(DefectSpec("hotspot_B7", "row_B7", "thermal_hotspot", 0.79),),
        extra_viewpoints=(
            ViewpointSpec(
                viewpoint_id="vp_safe_west_corridor",
                label="Safe west corridor waypoint",
                x=0.0,
                y=38.0,
                z=18.0,
                yaw_deg=0.0,
                asset_ids=(),
                standoff_bucket="far",
                suitable_modalities=("rgb", "thermal"),
                notes=("Use as a safe routing waypoint around the substation.",),
            ),
        )
        + EAST_RGB_VIEWPOINTS,
    ),
    "sparse_evidence_trap": SolarTaskSpec(
        task_id="sparse_evidence_trap",
        name="Sparse Evidence Trap",
        instruction=(
            "Do not submit after a single partial or unverified capture. Check the mission "
            "status, fill every missing row and anomaly requirement, then submit grounded evidence."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Reject premature evidence packs with missing rows.",
            "Confirm any detected anomaly with target-specific RGB evidence.",
        ),
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("grounding", "checklist", "anti-premature-submit"),
        hidden_defects=(DefectSpec("hotspot_B5", "row_B5", "thermal_hotspot", 0.83),),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "multi_anomaly_triage": SolarTaskSpec(
        task_id="multi_anomaly_triage",
        name="Multi-Anomaly Triage",
        instruction=(
            "Survey rows B4-B8, identify multiple thermal anomalies, and collect separate RGB "
            "confirmation for every affected row before final reporting."
        ),
        success_criteria=BASE_SUCCESS
        + ("Detect and report both thermal anomalies.", "Do not leave any anomaly without RGB evidence."),
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("anomaly", "triage", "multi-target"),
        hidden_defects=(
            DefectSpec("hotspot_B5", "row_B5", "thermal_hotspot", 0.84),
            DefectSpec("hotspot_B7", "row_B7", "thermal_hotspot", 0.89),
        ),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "closeup_resolution_challenge": SolarTaskSpec(
        task_id="closeup_resolution_challenge",
        name="Close-Up Resolution Challenge",
        instruction=(
            "Thermal overview is required, but anomaly confirmation must use a close, high-resolution "
            "RGB capture of the affected row. Use zoom or a close standoff if needed."
        ),
        success_criteria=BASE_SUCCESS
        + ("RGB anomaly evidence must satisfy a higher quality threshold.",),
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("resolution", "closeup", "rgb-confirmation"),
        hidden_defects=(DefectSpec("hotspot_B6", "row_B6", "thermal_hotspot", 0.92),),
        min_rgb_quality=0.68,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "edge_row_focus": SolarTaskSpec(
        task_id="edge_row_focus",
        name="Edge Row Focus",
        instruction=(
            "Inspect the full block and pay special attention to edge-row framing. A hidden "
            "edge-row anomaly may be near the frame boundary and still needs RGB confirmation."
        ),
        success_criteria=BASE_SUCCESS
        + ("Include edge rows B4 and B8 in accepted thermal evidence.",),
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("edge-framing", "anomaly"),
        hidden_defects=(DefectSpec("hotspot_B8", "row_B8", "thermal_hotspot", 0.87),),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "no_anomaly_clearance": SolarTaskSpec(
        task_id="no_anomaly_clearance",
        name="No-Anomaly Clearance",
        instruction=(
            "Perform a clean thermal clearance survey of rows B4-B8. If no anomaly is detected, "
            "submit a concise no-anomaly evidence pack without inventing findings."
        ),
        success_criteria=BASE_SUCCESS + ("Do not hallucinate anomalies in the final report.",),
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("clearance", "no-anomaly"),
        hidden_defects=(),
        rgb_closeup_for_anomalies=False,
    ),
    "obstacle_detour_inspection": SolarTaskSpec(
        task_id="obstacle_detour_inspection",
        name="Obstacle Detour Inspection",
        instruction=(
            "Complete the thermal survey while avoiding both the substation no-fly zone and a "
            "temporary crane obstacle. Use the visible corridor viewpoints to detour safely."
        ),
        success_criteria=BASE_SUCCESS + ("Avoid the temporary obstacle zone.",),
        public_constraints=BASE_CONSTRAINTS
        + ("A temporary crane obstacle blocks part of the usual west-to-overview route.",),
        task_tags=("safety", "obstacle", "routing"),
        hidden_defects=(DefectSpec("hotspot_B6", "row_B6", "thermal_hotspot", 0.8),),
        extra_zones=(
            ZoneSpec(
                zone_id="crane_obstacle_west",
                label="Temporary crane obstacle",
                min_x=12.0,
                min_y=18.0,
                max_x=24.0,
                max_y=26.0,
                zone_type="obstacle",
                reason="Temporary crane operating envelope",
            ),
        ),
        extra_viewpoints=(
            ViewpointSpec(
                viewpoint_id="vp_high_west_detour",
                label="High west detour corridor",
                x=0.0,
                y=38.0,
                z=18.0,
                yaw_deg=0.0,
                asset_ids=(),
                standoff_bucket="far",
                suitable_modalities=("rgb", "thermal"),
                notes=("Use to route around the crane obstacle.",),
            ),
        )
        + EAST_RGB_VIEWPOINTS,
    ),
    "privacy_zone_capture": SolarTaskSpec(
        task_id="privacy_zone_capture",
        name="Privacy-Zone Capture Discipline",
        instruction=(
            "Collect required inspection evidence while respecting a privacy-sensitive area. "
            "Flight may pass nearby, but captures from inside the privacy zone are invalid."
        ),
        success_criteria=BASE_SUCCESS + ("Do not capture imagery from inside the privacy zone.",),
        public_constraints=BASE_CONSTRAINTS
        + ("The privacy zone blocks image capture from the close west context area.",),
        task_tags=("privacy", "capture-safety"),
        hidden_defects=(DefectSpec("hotspot_B6", "row_B6", "thermal_hotspot", 0.86),),
        extra_zones=(
            ZoneSpec(
                zone_id="neighbor_privacy_yard",
                label="Neighbor privacy zone",
                min_x=28.0,
                min_y=8.0,
                max_x=34.0,
                max_y=20.0,
                zone_type="privacy",
                constraint_level="hard",
                reason="No image capture from this area",
            ),
        ),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "return_home_compliance": SolarTaskSpec(
        task_id="return_home_compliance",
        name="Return-Home Compliance",
        instruction=(
            "A report is only acceptable after collecting evidence, returning home, and landing. "
            "Do not submit from the inspection area even if the photos look complete."
        ),
        success_criteria=BASE_SUCCESS + ("Submit only after returned_home and landed are true.",),
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("procedure", "return-home"),
        hidden_defects=(DefectSpec("hotspot_B7", "row_B7", "thermal_hotspot", 0.77),),
        min_report_grounding_score=0.65,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "limited_steps_rapid_survey": SolarTaskSpec(
        task_id="limited_steps_rapid_survey",
        name="Limited-Steps Rapid Survey",
        instruction=(
            "Complete the thermal survey under a tight step budget. Prefer a compact route and "
            "avoid unnecessary inspections, hovering, or redundant captures."
        ),
        success_criteria=BASE_SUCCESS + ("Finish within the reduced step limit.",),
        public_constraints=BASE_CONSTRAINTS + ("The episode has a reduced step budget.",),
        task_tags=("steps", "efficiency"),
        hidden_defects=(),
        max_steps=14,
        rgb_closeup_for_anomalies=False,
    ),
    "report_grounding_audit": SolarTaskSpec(
        task_id="report_grounding_audit",
        name="Report Grounding Audit",
        instruction=(
            "Produce a final evidence pack that is auditable: every cited photo ID must be real, "
            "useful, and connected to the rows or anomalies claimed in the summary."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Mention detected anomaly IDs in the report.",
            "Cite the thermal overview and the matching RGB anomaly evidence.",
        ),
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("report-grounding", "audit", "anomaly"),
        hidden_defects=(DefectSpec("hotspot_B6", "row_B6", "thermal_hotspot", 0.9),),
        min_report_grounding_score=0.75,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
}


def get_solar_task(task_id: str | None) -> SolarTaskSpec:
    """Return a task spec by id, falling back to the default task."""

    normalized = DEFAULT_TASK if task_id is None else task_id
    if normalized not in SOLAR_TASKS:
        supported = ", ".join(sorted(SOLAR_TASKS))
        raise ValueError(f"unsupported solar task {normalized!r}; supported tasks: {supported}")
    return SOLAR_TASKS[normalized]
