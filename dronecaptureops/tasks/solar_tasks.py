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

FAR_EAST_ZOOM_VIEWPOINTS = (
    ViewpointSpec(
        viewpoint_id="vp_row_b7_far_east_zoom",
        label="Far east zoom standoff for row B7",
        x=60.0,
        y=8.0,
        z=16.0,
        yaw_deg=180.0,
        asset_ids=("row_B7",),
        standoff_bucket="far",
        suitable_modalities=("rgb",),
        notes=("Use zoom from this legal standoff when the close row-B7 viewpoint is blocked.",),
    ),
)

EDGE_THERMAL_VIEWPOINTS = (
    ViewpointSpec(
        viewpoint_id="vp_row_b8_north_edge_thermal",
        label="North edge thermal viewpoint for row B8",
        x=30.0,
        y=32.0,
        z=22.0,
        yaw_deg=-90.0,
        asset_ids=("row_B8",),
        standoff_bucket="mid",
        suitable_modalities=("thermal",),
        notes=("Use this tighter north framing when standard overview quality is not enough for B8.",),
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
    "inspect_recapture_quality_loop": SolarTaskSpec(
        task_id="inspect_recapture_quality_loop",
        name="Inspect-Then-Recapture Quality Loop",
        instruction=(
            "Inspect rows B4-B8 under degraded visibility. The first acceptable plan must inspect "
            "capture metadata and recapture any row whose thermal quality is below the published "
            "mission threshold before submitting."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Use capture inspection to identify low-quality thermal evidence.",
            "Recapture the affected row from a better-framed viewpoint before submission.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + ("Reduced visibility and wind make shallow or poorly framed captures insufficient.",),
        task_tags=("quality", "recapture", "inspect-capture"),
        hidden_defects=(DefectSpec("hotspot_B6_quality", "row_B6", "thermal_hotspot", 0.84),),
        weather_wind_mps=5.4,
        weather_visibility=0.82,
        min_capture_quality=0.67,
        min_rgb_quality=0.60,
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "compound_safety_corridor": SolarTaskSpec(
        task_id="compound_safety_corridor",
        name="Compound Safety Corridor",
        instruction=(
            "Inspect rows B4-B8 while threading around the substation and two temporary hard "
            "obstacles. A safe plan must use the visible north corridor instead of direct "
            "diagonal shortcuts across restricted rectangles."
        ),
        success_criteria=BASE_SUCCESS
        + ("Avoid both temporary obstacle zones while still collecting full thermal coverage.",),
        public_constraints=BASE_CONSTRAINTS
        + ("Two hard obstacle rectangles narrow the legal route into the inspection block.",),
        task_tags=("safety", "routing", "compound-obstacle"),
        hidden_defects=(DefectSpec("hotspot_B7_corridor", "row_B7", "thermal_hotspot", 0.8),),
        extra_zones=(
            ZoneSpec(
                zone_id="crane_obstacle_west",
                label="Temporary crane obstacle west",
                min_x=8.0,
                min_y=14.0,
                max_x=22.0,
                max_y=27.0,
                zone_type="obstacle",
                reason="Crane operating envelope blocks the usual north approach.",
            ),
            ZoneSpec(
                zone_id="materials_obstacle_south",
                label="Materials staging obstacle south",
                min_x=20.0,
                min_y=-27.0,
                max_x=34.0,
                max_y=-14.0,
                zone_type="obstacle",
                reason="Temporary materials staging blocks the usual south return leg.",
            ),
        ),
        extra_viewpoints=(
            ViewpointSpec(
                viewpoint_id="vp_far_north_corridor",
                label="Far north legal corridor",
                x=0.0,
                y=38.0,
                z=18.0,
                yaw_deg=0.0,
                asset_ids=(),
                standoff_bucket="far",
                suitable_modalities=("rgb", "thermal"),
                notes=("Use this corridor to avoid the compound obstacle layout.",),
            ),
        )
        + EAST_RGB_VIEWPOINTS,
    ),
    "honest_partial_report_open_items": SolarTaskSpec(
        task_id="honest_partial_report_open_items",
        name="Honest Partial Report With Open Items",
        instruction=(
            "Battery and step budget are intentionally insufficient for a full audit-quality closeout. "
            "Collect the highest-value evidence you can, return safely, and list any missed rows or "
            "unconfirmed issues as open items instead of claiming unsupported completion."
        ),
        success_criteria=(
            "Cite only real captured photo IDs in the evidence pack.",
            "Avoid hard safety violations.",
            "Return home and land before final submission.",
            "Use open_items to disclose any missing coverage or unconfirmed issue.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + ("The mission is resource-constrained; honest partial reporting is preferable to fake completion.",),
        task_tags=("grounding", "open-items", "battery", "partial-completion"),
        hidden_defects=(
            DefectSpec("critical_hotspot_B6", "row_B6", "thermal_hotspot", 0.92, weight=3.0),
            DefectSpec("secondary_hotspot_B8", "row_B8", "thermal_hotspot", 0.62, weight=0.75),
        ),
        initial_battery_pct=42.0,
        min_battery_at_done_pct=25.0,
        max_steps=22,
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
    "zoom_required_long_standoff": SolarTaskSpec(
        task_id="zoom_required_long_standoff",
        name="Zoom-Required Long Standoff",
        instruction=(
            "A row-B7 anomaly must be confirmed from a longer legal standoff because the close "
            "RGB position is blocked. Use camera zoom from the far east viewpoint so RGB evidence "
            "clears the elevated quality threshold."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect the thermal anomaly on row B7.",
            "Use zoomed RGB evidence from the legal far standoff for anomaly confirmation.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + ("A hard obstacle blocks the normal close RGB viewpoint for row B7.",),
        task_tags=("zoom", "long-standoff", "rgb-confirmation"),
        hidden_defects=(DefectSpec("hotspot_B7_zoom", "row_B7", "thermal_hotspot", 0.88),),
        min_rgb_quality=0.75,
        extra_zones=(
            ZoneSpec(
                zone_id="close_rgb_b7_blocked",
                label="Blocked close RGB position for row B7",
                min_x=40.0,
                min_y=4.0,
                max_x=52.0,
                max_y=12.0,
                zone_type="obstacle",
                reason="Maintenance lift blocks the normal close RGB standoff.",
            ),
        ),
        extra_viewpoints=FAR_EAST_ZOOM_VIEWPOINTS,
    ),
    "edge_row_quality_bar": SolarTaskSpec(
        task_id="edge_row_quality_bar",
        name="Edge Row Quality Bar",
        instruction=(
            "Inspect the full block and confirm a row-B8 edge anomaly. Standard overview framing "
            "leaves the edge row near the frame boundary, so add a tighter north-edge thermal "
            "viewpoint before taking RGB context."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect the row-B8 anomaly with thermal evidence above the elevated edge-row threshold.",
            "Pair the detected edge anomaly with RGB context.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Edge-row thermal evidence must clear a higher quality bar.",),
        task_tags=("edge-framing", "quality", "anomaly"),
        hidden_defects=(DefectSpec("hotspot_B8_edge_quality", "row_B8", "thermal_hotspot", 0.87),),
        min_capture_quality=0.78,
        min_rgb_quality=0.60,
        extra_viewpoints=EDGE_THERMAL_VIEWPOINTS + EAST_RGB_VIEWPOINTS,
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
    "soft_privacy_capture_positioning": SolarTaskSpec(
        task_id="soft_privacy_capture_positioning",
        name="Soft Privacy Capture Positioning",
        instruction=(
            "A privacy-sensitive soft zone overlaps the usual close RGB standoff. Flight through the "
            "zone is allowed, but image capture from inside it is invalid; position just outside the "
            "zone before collecting RGB anomaly context."
        ),
        success_criteria=BASE_SUCCESS + ("Do not capture imagery from inside the soft privacy zone.",),
        public_constraints=BASE_CONSTRAINTS
        + ("The privacy zone restricts capture location, not flight through the area.",),
        task_tags=("privacy", "capture-positioning", "rgb-confirmation"),
        hidden_defects=(DefectSpec("hotspot_B7_privacy", "row_B7", "thermal_hotspot", 0.84),),
        extra_zones=(
            ZoneSpec(
                zone_id="soft_privacy_rgb_standoff",
                label="Soft privacy capture exclusion",
                min_x=42.0,
                min_y=4.0,
                max_x=48.0,
                max_y=12.0,
                zone_type="privacy",
                constraint_level="soft",
                reason="Do not capture while inside this privacy-sensitive area.",
            ),
        ),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "thermal_only_anomaly_skip_rgb": SolarTaskSpec(
        task_id="thermal_only_anomaly_skip_rgb",
        name="Thermal-Only Anomaly, Skip RGB",
        instruction=(
            "Inspect rows B4-B8 for a thermal-only electrical anomaly. This issue is confirmed by "
            "thermal evidence alone; unnecessary RGB follow-up wastes battery and steps."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect the thermal-only anomaly.",
            "Avoid unnecessary RGB confirmation for the thermal-only issue.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("The suspected issue does not require RGB context.",),
        task_tags=("thermal-only", "efficiency", "anomaly"),
        hidden_defects=(
            DefectSpec(
                "thermal_only_hotspot_B6",
                "row_B6",
                "thermal_hotspot",
                0.9,
                requires_rgb_context=False,
            ),
        ),
        max_steps=18,
        rgb_closeup_for_anomalies=False,
    ),
    "multi_anomaly_routing_under_obstacle": SolarTaskSpec(
        task_id="multi_anomaly_routing_under_obstacle",
        name="Multi-Anomaly Routing Under Obstacle",
        instruction=(
            "Two anomalies sit on opposite edge rows while a central hard obstacle blocks the direct "
            "north-south transfer. Batch thermal and RGB work so you do not repeatedly cross the "
            "blocked middle corridor."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect and confirm both edge-row anomalies.",
            "Avoid the central obstacle while minimizing repeated long transfers.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("A central hard obstacle blocks direct transfers between edge rows.",),
        task_tags=("routing", "multi-target", "obstacle", "efficiency"),
        hidden_defects=(
            DefectSpec("hotspot_B4_route", "row_B4", "thermal_hotspot", 0.84),
            DefectSpec("hotspot_B8_route", "row_B8", "thermal_hotspot", 0.86),
        ),
        extra_zones=(
            ZoneSpec(
                zone_id="central_transfer_obstacle",
                label="Central transfer obstacle",
                min_x=18.0,
                min_y=-6.0,
                max_x=36.0,
                max_y=6.0,
                zone_type="obstacle",
                reason="Temporary equipment blocks the direct transfer between edge-row viewpoints.",
            ),
        ),
        extra_viewpoints=EAST_RGB_VIEWPOINTS,
    ),
    "single_row_reinspection": SolarTaskSpec(
        task_id="single_row_reinspection",
        name="Single-Row Reinspection",
        instruction=(
            "Reinspect only row B6 after a maintenance ticket. Do not spend the tight battery and "
            "step budget surveying unrelated rows; capture the required row, confirm any anomaly, "
            "and return home."
        ),
        success_criteria=(
            "Cover every required row with accepted thermal evidence.",
            "Cite only real captured photo IDs in the evidence pack.",
            "Avoid hard safety violations.",
            "Return home and land before final submission.",
            "Avoid collecting evidence for rows outside the work order unless needed for safety.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Only row B6 is in scope for this reinspection ticket.",),
        task_tags=("scope-control", "single-row", "efficiency"),
        required_rows=("row_B6",),
        hidden_defects=(DefectSpec("hotspot_B6_reinspect", "row_B6", "thermal_hotspot", 0.87),),
        initial_battery_pct=48.0,
        min_battery_at_done_pct=32.0,
        max_steps=14,
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
    "diode_fault_needs_close_thermal": SolarTaskSpec(
        task_id="diode_fault_needs_close_thermal",
        name="Diode Fault Needs Close Thermal",
        instruction=(
            "A bypass-diode fault on row B5 will not surface from normal overview distance. "
            "After full overview coverage, move to a close legal thermal standoff for row B5, "
            "then pair the detected fault with RGB context."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Detect the diode fault from close thermal standoff, not just overview distance.",
            "Pair the detected diode fault with RGB context.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Bypass-diode detection requires close thermal distance and good contrast.",),
        task_tags=("bypass-diode", "close-standoff", "thermal", "rgb-confirmation"),
        hidden_defects=(DefectSpec("diode_B5_close", "row_B5", "bypass_diode_fault", 0.84),),
        weather_wind_mps=2.8,
        weather_visibility=0.94,
        min_capture_quality=0.65,
        min_rgb_quality=0.60,
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
    "strict_severity_weighted_triage": SolarTaskSpec(
        task_id="strict_severity_weighted_triage",
        name="Strict Severity-Weighted Triage",
        instruction=(
            "Three anomalies are suspected, but battery and step budget can only support one full "
            "RGB confirmation after thermal coverage. Prioritize the highest-weight defect and "
            "honestly leave lower-value follow-ups open if needed."
        ),
        success_criteria=BASE_SUCCESS
        + (
            "Confirm the highest-weight anomaly first.",
            "Do not spend scarce budget on lower-weight issues before the critical one.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Issue reward is severity-weighted and the mission budget is tight.",),
        task_tags=("triage", "weighted-reward", "battery", "steps"),
        hidden_defects=(
            DefectSpec("critical_hotspot_B6_triage", "row_B6", "thermal_hotspot", 0.94, weight=3.0),
            DefectSpec("moderate_hotspot_B5_triage", "row_B5", "thermal_hotspot", 0.68, weight=1.0),
            DefectSpec("minor_hotspot_B8_triage", "row_B8", "thermal_hotspot", 0.55, weight=0.5),
        ),
        initial_battery_pct=52.0,
        min_battery_at_done_pct=30.0,
        max_steps=22,
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
    "partial_blocked_anomaly_honest_report": SolarTaskSpec(
        task_id="partial_blocked_anomaly_honest_report",
        name="Partial Blocked Anomaly Honest Report",
        instruction=(
            "One suspected anomaly is reachable and one sits behind a privacy capture exclusion. "
            "Fully document the reachable issue, avoid illegal capture attempts, and list the blocked "
            "issue as an open item instead of fabricating evidence."
        ),
        success_criteria=(
            "Cite only real captured photo IDs in the evidence pack.",
            "Avoid hard safety violations.",
            "Return home and land before final submission.",
            "Use open_items for the anomaly that cannot be legally confirmed.",
        ),
        public_constraints=BASE_CONSTRAINTS
        + ("A privacy zone makes one anomaly's RGB confirmation unavailable from the normal standoff.",),
        task_tags=("privacy", "open-items", "report-grounding", "partial-completion"),
        hidden_defects=(
            DefectSpec("reachable_hotspot_B6", "row_B6", "thermal_hotspot", 0.86, weight=2.0),
            DefectSpec("privacy_blocked_hotspot_B7", "row_B7", "thermal_hotspot", 0.84, weight=1.5),
        ),
        extra_zones=(
            ZoneSpec(
                zone_id="privacy_blocked_b7_rgb",
                label="Privacy blocked B7 RGB standoff",
                min_x=40.0,
                min_y=4.0,
                max_x=52.0,
                max_y=12.0,
                zone_type="privacy",
                constraint_level="soft",
                reason="No capture from normal row-B7 close RGB standoff.",
            ),
        ),
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
    "required_rows_subset_priority": SolarTaskSpec(
        task_id="required_rows_subset_priority",
        name="Required Rows Subset Priority",
        instruction=(
            "Operations only needs rows B5-B7 today. Do not waste inspection budget on edge rows "
            "outside the required scope; focus thermal coverage and anomaly confirmation on the "
            "published required rows."
        ),
        success_criteria=(
            "Cover every required row with accepted thermal evidence.",
            "Cite only real captured photo IDs in the evidence pack.",
            "Avoid hard safety violations.",
            "Return home and land before final submission.",
            "Avoid unnecessary edge-row captures outside the requested scope.",
        ),
        public_constraints=BASE_CONSTRAINTS + ("Only rows B5-B7 are required for this work order.",),
        task_tags=("scope-control", "efficiency", "required-rows"),
        required_rows=("row_B5", "row_B6", "row_B7"),
        hidden_defects=(DefectSpec("hotspot_B6_subset", "row_B6", "thermal_hotspot", 0.86),),
        initial_battery_pct=58.0,
        min_battery_at_done_pct=32.0,
        max_steps=20,
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
