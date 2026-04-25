"""Task-conditioned solar inspection missions."""

from __future__ import annotations

from dataclasses import dataclass, field

from dronecaptureops.core.constants import DEFAULT_MAX_STEPS, DEFAULT_TASK


@dataclass(frozen=True)
class DefectSpec:
    """Deterministic hidden defect for a solar task.

    Extra fields mirror `HiddenDefect` so a task can declare false-positive
    or thermal-only defects (e.g. glare artifacts, no-RGB-needed soiling)
    without the camera/verifier expecting RGB confirmation or counting them
    as required issues.
    """

    defect_id: str
    target_id: str
    defect_type: str
    severity: float
    weight: float = 2.0
    counts_for_issue_reward: bool = True
    requires_rgb_context: bool = True


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
    "string_outage_survey": SolarTaskSpec(
        task_id="string_outage_survey",
        name="String Outage Survey",
        instruction=(
            "Survey rows B4-B8. A whole-row underperformance (string-level outage) is suspected on one "
            "row. Capture full thermal coverage, then collect RGB context for the affected row and cite "
            "both in the evidence pack."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect the string-level thermal anomaly.",
            "Pair the anomaly with RGB context showing the same row.",
        ),
        public_constraints=BASE_CONSTRAINTS,
        task_tags=("anomaly", "string-outage", "rgb-confirmation"),
        hidden_defects=(DefectSpec("string_outage_B6", "row_B6", "thermal_hotspot", 0.86),),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "pid_multi_row_pattern": SolarTaskSpec(
        task_id="pid_multi_row_pattern",
        name="PID Multi-Row Pattern",
        instruction=(
            "Inspect rows B4-B8 with cleanly framed thermal evidence. Subtle PID-style degradation may "
            "appear on multiple adjacent rows; capture each affected row and pair every detected anomaly "
            "with target-specific RGB context before submitting."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect every adjacent-row PID anomaly.",
            "Pair every detected anomaly with RGB context.",
            "Thermal evidence must clear the elevated quality threshold.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Subtle anomalies require higher thermal quality.",),
        task_tags=("anomaly", "pid", "multi-target", "quality"),
        hidden_defects=(
            DefectSpec("pid_B5", "row_B5", "thermal_hotspot", 0.55),
            DefectSpec("pid_B6", "row_B6", "thermal_hotspot", 0.5),
            DefectSpec("pid_B7", "row_B7", "thermal_hotspot", 0.6),
        ),
        min_capture_quality=0.62,
        min_rgb_quality=0.6,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "cracked_glass_closeup": SolarTaskSpec(
        task_id="cracked_glass_closeup",
        name="Cracked Glass Close-Up",
        instruction=(
            "Inspect rows B4-B8. A suspected glass crack on one row needs both a thermal anomaly "
            "frame and a high-resolution RGB close-up that shows the physical damage."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect the suspected anomaly on thermal.",
            "Confirm the damage with a high-quality RGB close-up of the affected row.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Anomaly RGB evidence must clear an elevated quality bar.",),
        task_tags=("anomaly", "rgb-confirmation", "closeup", "physical-damage"),
        hidden_defects=(DefectSpec("cracked_glass_B5", "row_B5", "thermal_hotspot", 0.83),),
        min_rgb_quality=0.72,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "bird_soiling_explanation": SolarTaskSpec(
        task_id="bird_soiling_explanation",
        name="Bird Soiling Explanation",
        instruction=(
            "Survey rows B4-B8. A localized warm spot is suspected to be caused by surface soiling "
            "(droppings or dust). Capture thermal evidence and an RGB close-up that explains the "
            "physical cause, then submit a grounded report."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect the soiling-related thermal anomaly.",
            "Provide RGB evidence showing the physical cause.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Calm conditions are required for soiling detection.",),
        task_tags=("anomaly", "soiling", "rgb-confirmation", "explanation"),
        hidden_defects=(DefectSpec("soiling_B6", "row_B6", "soiling_heating", 0.55),),
        weather_wind_mps=2.0,
        weather_visibility=0.95,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "vegetation_edge_encroachment": SolarTaskSpec(
        task_id="vegetation_edge_encroachment",
        name="Vegetation Edge Encroachment",
        instruction=(
            "Inspect rows B4-B8 with attention to the edge rows. Vegetation may be encroaching on "
            "an edge row and creating partial shading; capture an oblique-angle thermal that shows "
            "the shadow and an RGB context shot for the same edge row."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Include edge rows B4 and B8 in accepted thermal evidence.",
            "Detect the vegetation-shadow anomaly on the affected edge row.",
            "Pair the anomaly with RGB context showing vegetation.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Vegetation shadows require an oblique gimbal angle, not pure overhead.",),
        task_tags=("anomaly", "vegetation", "edge-framing", "rgb-confirmation"),
        hidden_defects=(DefectSpec("veg_shadow_B8", "row_B8", "vegetation_shadow", 0.55),),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "substation_adjacency_caution": SolarTaskSpec(
        task_id="substation_adjacency_caution",
        name="Substation Adjacency Caution",
        instruction=(
            "A suspected hotspot is near the substation perimeter. Maintain safe standoff from the "
            "substation no-fly zone and an additional safety buffer just outside it, then capture "
            "thermal coverage and an RGB close-up of the affected row from a legal viewpoint."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Do not enter the substation no-fly zone or its safety buffer.",
            "Capture the affected row from a legal close viewpoint.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + (
            "An additional safety buffer extends north of the substation no-fly zone.",
            "All hard zones must be respected by both flight path and viewpoint.",
        ),
        task_tags=("safety", "no-fly", "anomaly", "rgb-confirmation"),
        hidden_defects=(DefectSpec("hotspot_B5_adj", "row_B5", "thermal_hotspot", 0.85),),
        extra_zones=(
            ZoneSpec(
                zone_id="substation_safety_buffer_north",
                label="Substation safety buffer (north)",
                min_x=14.0,
                min_y=6.0,
                max_x=24.0,
                max_y=12.0,
                zone_type="no_fly",
                constraint_level="hard",
                reason="Safety buffer above substation no-fly zone.",
            ),
        ),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "low_contrast_recapture": SolarTaskSpec(
        task_id="low_contrast_recapture",
        name="Low Contrast Recapture",
        instruction=(
            "Inspect rows B4-B8 in low-contrast conditions (reduced visibility and elevated wind). "
            "Use inspect_capture to evaluate quality and recapture any thermal frame that does not "
            "clear the elevated quality bar before submitting evidence."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Recapture any low-quality thermal frame before submission.",
            "Pair the detected anomaly with RGB context.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + ("Reduced visibility lowers thermal contrast.", "Elevated wind reduces blur stability."),
        task_tags=("weather", "quality", "recovery", "anomaly"),
        hidden_defects=(DefectSpec("hotspot_B6_lc", "row_B6", "thermal_hotspot", 0.82),),
        weather_wind_mps=5.0,
        weather_visibility=0.78,
        min_capture_quality=0.62,
        min_rgb_quality=0.6,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "true_false_anomaly_discrimination": SolarTaskSpec(
        task_id="true_false_anomaly_discrimination",
        name="True/False Anomaly Discrimination",
        instruction=(
            "A real thermal anomaly is suspected on one row, but glare-induced false positives may "
            "also appear from shallow gimbal angles. Use steep-pitch thermal coverage to confirm the "
            "real anomaly with RGB context, and do not report any glare artifact as a real issue."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect and confirm the real thermal anomaly.",
            "Do not include glare-only artifacts in the issues list.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + ("Shallow gimbal angles increase glare risk and can produce false thermal artifacts.",),
        task_tags=("anomaly", "false-positive", "discrimination", "report-grounding"),
        hidden_defects=(
            DefectSpec("hotspot_B6_real", "row_B6", "thermal_hotspot", 0.85),
            DefectSpec(
                "glare_artifact_B7_fp",
                "row_B7",
                "false_thermal_artifact",
                0.4,
                weight=0.0,
                counts_for_issue_reward=False,
                requires_rgb_context=False,
            ),
        ),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "permanent_occlusion_coverage": SolarTaskSpec(
        task_id="permanent_occlusion_coverage",
        name="Permanent Occlusion Coverage",
        instruction=(
            "A permanent maintenance vehicle blocks the north overview corridor for the entire mission. "
            "Use the south overview viewpoint and any safe alternates to cover all rows B4-B8 and "
            "capture the suspected anomaly without crossing the obstacle zone."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Cover every required row using the south overview viewpoint or a safe alternate.",
            "Avoid the permanent maintenance vehicle obstacle zone.",
            "Pair the detected anomaly with RGB context.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + ("The north overview corridor is permanently blocked by a maintenance vehicle.",),
        task_tags=("safety", "obstacle", "coverage", "anomaly"),
        hidden_defects=(DefectSpec("hotspot_B5_occ", "row_B5", "thermal_hotspot", 0.84),),
        extra_zones=(
            ZoneSpec(
                zone_id="maintenance_vehicle_north",
                label="Maintenance vehicle (north overview)",
                min_x=22.0,
                min_y=18.0,
                max_x=38.0,
                max_y=30.0,
                max_altitude_m=40.0,
                zone_type="obstacle",
                constraint_level="hard",
                reason="Maintenance vehicle blocks north overview corridor.",
            ),
        ),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "prioritized_triage_under_constraint": SolarTaskSpec(
        task_id="prioritized_triage_under_constraint",
        name="Prioritized Triage Under Constraint",
        instruction=(
            "Two anomalies are suspected with different severities. Battery and step budget are tight, "
            "so prioritize the high-severity anomaly with RGB confirmation and at least cover the "
            "remaining rows on thermal before returning home."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Confirm the high-severity anomaly with RGB context.",
            "Cover every required row on thermal even under the tight budget.",
            "Finish with at least the stricter battery reserve.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Battery and step budgets are tight.",),
        task_tags=("anomaly", "triage", "battery", "steps"),
        hidden_defects=(
            DefectSpec("severe_hotspot_B6", "row_B6", "thermal_hotspot", 0.92, weight=3.0),
            DefectSpec("mild_hotspot_B8", "row_B8", "thermal_hotspot", 0.55, weight=1.0),
        ),
        initial_battery_pct=50.0,
        min_battery_at_done_pct=30.0,
        max_steps=24,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "capture_efficiency_discipline": SolarTaskSpec(
        task_id="capture_efficiency_discipline",
        name="Capture Efficiency Discipline",
        instruction=(
            "Complete a clean thermal coverage of rows B4-B8 with the minimum number of captures. "
            "Avoid redundant captures: the same sensor framing the same rows twice is penalized."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Cover every required row with a minimal capture count.",
            "Do not collect redundant captures of the same rows.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + (
            "Step budget is tight.",
            "Battery reserve is reduced.",
            "Redundant duplicate captures are penalized.",
        ),
        task_tags=("efficiency", "steps", "battery", "no-redundant-captures"),
        hidden_defects=(),
        initial_battery_pct=55.0,
        min_battery_at_done_pct=30.0,
        max_steps=16,
        rgb_closeup_for_anomalies=False,
    ),
    "boundary_aware_closeup": SolarTaskSpec(
        task_id="boundary_aware_closeup",
        name="Boundary-Aware Close-Up",
        instruction=(
            "A high-severity anomaly is suspected on a row close to the substation no-fly boundary. "
            "Position carefully outside the no-fly zone, use zoom or a tight legal close standoff, and "
            "capture both a thermal and an RGB close-up of the affected row."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Capture a high-quality close-up of the affected row from a legal viewpoint.",
            "Do not enter the substation no-fly zone.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + ("Anomaly RGB and thermal evidence must both clear an elevated quality bar.",),
        task_tags=("anomaly", "no-fly", "closeup", "gimbal"),
        hidden_defects=(DefectSpec("hotspot_B5_boundary", "row_B5", "thermal_hotspot", 0.88),),
        min_capture_quality=0.62,
        min_rgb_quality=0.7,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "no_defect_with_glare_artifact": SolarTaskSpec(
        task_id="no_defect_with_glare_artifact",
        name="No-Defect With Glare Artifact",
        instruction=(
            "Perform a clean clearance survey of rows B4-B8. Glare artifacts can appear at shallow "
            "gimbal angles. Verify any anomaly carefully and do not report glare-only artifacts as "
            "real issues."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Do not report glare-only artifacts as real issues.",
            "Submit a concise no-anomaly evidence pack with full row coverage.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + ("Shallow gimbal angles increase glare risk and can produce false thermal artifacts.",),
        task_tags=("clearance", "no-anomaly", "false-positive", "report-grounding"),
        hidden_defects=(
            DefectSpec(
                "glare_artifact_B6_fp",
                "row_B6",
                "false_thermal_artifact",
                0.4,
                weight=0.0,
                counts_for_issue_reward=False,
                requires_rgb_context=False,
            ),
        ),
        rgb_closeup_for_anomalies=False,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "adaptive_battery_reserve": SolarTaskSpec(
        task_id="adaptive_battery_reserve",
        name="Adaptive Battery Reserve",
        instruction=(
            "Two anomalies are suspected on different rows. Battery is moderate and the reserve "
            "requirement is stricter than usual. Choose dynamically whether to investigate both or "
            "return early once coverage and battery margin are both safe."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Cover every required row on thermal.",
            "Finish with at least the stricter battery reserve.",
            "Do not violate the return-home requirement under battery pressure.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Stricter return-home battery reserve is required.",),
        task_tags=("battery", "anomaly", "tradeoff", "return-home"),
        hidden_defects=(
            DefectSpec("hotspot_B6_easy", "row_B6", "thermal_hotspot", 0.86),
            DefectSpec("hotspot_B4_edge", "row_B4", "thermal_hotspot", 0.78),
        ),
        initial_battery_pct=60.0,
        min_battery_at_done_pct=35.0,
        max_steps=28,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "audit_grade_strict_grounding": SolarTaskSpec(
        task_id="audit_grade_strict_grounding",
        name="Audit-Grade Strict Grounding",
        instruction=(
            "Produce an audit-grade evidence pack: cite separate thermal coverage photos and a "
            "matching RGB close-up for each anomaly, and avoid any unsupported or duplicated claim. "
            "The grounding bar is stricter than usual."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Cite the thermal overview and a matching RGB close-up for each anomaly.",
            "Do not include unsupported or duplicate claims in the report.",
            "Final grounding score must clear the stricter threshold.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Final report grounding threshold is stricter than usual.",),
        task_tags=("report-grounding", "audit", "anomaly", "multi-target"),
        hidden_defects=(
            DefectSpec("hotspot_B5_audit", "row_B5", "thermal_hotspot", 0.86),
            DefectSpec("hotspot_B7_audit", "row_B7", "thermal_hotspot", 0.88),
        ),
        min_report_grounding_score=0.85,
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
