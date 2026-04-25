from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import EvidenceReport
from dronecaptureops.core.models import RawDroneAction
from dronecaptureops.rewards.verifiers import compute_integrity_gate, compute_issue_capture


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def capture_thermal_overview(env: DroneCaptureOpsEnvironment):
    """Two staggered thermal passes covering rows B4-B8 from north + south."""

    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=22, yaw_deg=-90, speed_mps=5))
    env.step(act("set_camera_source", source="thermal"))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    env.step(act("capture_thermal", label="overview B6-B8"))
    env.step(act("fly_to_viewpoint", x=30, y=-24, z=22, yaw_deg=90, speed_mps=5))
    env.step(act("set_gimbal", pitch_deg=-56, yaw_deg=0))
    return env.step(act("capture_thermal", label="overview B4-B6"))


def complete_capture_flow(env: DroneCaptureOpsEnvironment):
    """Full thermal coverage + RGB anomaly context + return home + land."""

    obs = capture_thermal_overview(env)
    for capture in obs.capture_log:
        if capture.sensor == "thermal":
            env.step(act("inspect_capture", photo_id=capture.photo_id))
    # We are at the south overview, so do the south RGB close-up first.
    env.step(act("fly_to_viewpoint", x=30, y=-24, z=14, yaw_deg=90, speed_mps=4))
    env.step(act("set_camera_source", source="rgb"))
    env.step(act("set_gimbal", pitch_deg=-45, yaw_deg=0))
    env.step(act("capture_rgb", label="rgb close-up south"))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=14, yaw_deg=-90, speed_mps=4))
    env.step(act("set_gimbal", pitch_deg=-45, yaw_deg=0))
    obs = env.step(act("capture_rgb", label="rgb close-up north"))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("return_home"))
    return env.step(act("land"))


def test_reward_breakdown_has_expected_shape():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7)

    keys = set(obs.reward_breakdown.model_dump())

    assert {
        "evidence_success",
        "required_coverage",
        "issue_capture",
        "operational_efficiency",
        "grounded_report",
        "process_reward",
        "integrity_gate",
        "value_per_photo",
        "target_coverage",
        "capture_quality",
        "defect_visibility",
        "checklist_completion",
        "route_efficiency",
        "battery_management",
        "safety_compliance",
        "report_grounding",
        "total",
    } <= keys


def test_submit_report_rejects_fake_photo_id():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step(act("submit_evidence_pack", summary="fake", photo_ids=["IMG-T-999"], findings=[]))

    assert obs.done is True
    assert obs.action_result["accepted"] is False
    assert obs.reward_breakdown.report_grounding < 0.5
    assert obs.reward_breakdown.integrity_gate == 0.2


def test_report_without_evidence_is_capped():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step(act("submit_evidence_pack", summary="complete", photo_ids=[], findings=[]))

    assert obs.action_result["accepted"] is False
    assert obs.reward_breakdown.integrity_gate == 0.2
    assert obs.reward_breakdown.total <= 0.2


def test_wrong_sensor_does_not_satisfy_thermal_coverage():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=18, yaw_deg=-90, speed_mps=5))
    env.step(act("set_gimbal", pitch_deg=-60, yaw_deg=0))

    obs = env.step(act("capture_rgb", label="wrong sensor overview"))

    assert obs.last_capture.sensor == "rgb"
    assert obs.reward_breakdown.required_coverage == 0.0
    assert obs.reward_breakdown.target_coverage == 0.0


def test_thermal_only_anomaly_gets_partial_issue_credit():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = capture_thermal_overview(env)

    assert obs.reward_breakdown.issue_capture == 0.6
    assert obs.reward_breakdown.defect_visibility == 0.6


def test_thermal_plus_rgb_anomaly_gets_full_issue_credit():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = complete_capture_flow(env)

    assert obs.reward_breakdown.issue_capture == 1.0
    assert set(obs.checklist_status.anomalies_detected) <= set(obs.checklist_status.anomaly_rgb_pairs)


def test_redundant_photo_adds_penalty():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    capture_thermal_overview(env)

    obs = env.step(act("capture_thermal", label="duplicate overview"))

    assert obs.reward_breakdown.penalties >= 0.02


def test_early_submission_cannot_score_highly():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step(act("submit_evidence_pack", summary="done", photo_ids=[], findings=[]))

    assert obs.reward_breakdown.evidence_success < 0.5
    assert obs.reward_breakdown.total <= 0.2


def test_backward_compatible_report_payload_can_complete_mission():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    obs = complete_capture_flow(env)
    photo_ids = [capture.photo_id for capture in obs.capture_log]
    defect_ids = [defect.defect_id for defect in env.debug_world.hidden_defects]

    obs = env.step(
        act(
            "submit_evidence_pack",
            summary=f"Rows inspected. Issues: {' '.join(defect_ids)}. Returned home with battery reserve.",
            photo_ids=photo_ids,
            findings=[{"finding": defect_id, "photo_ids": photo_ids} for defect_id in defect_ids],
            safety_notes=["Returned home with battery reserve."],
        )
    )

    assert obs.action_result["accepted"] is True
    assert obs.reward_breakdown.integrity_gate == 1.0
    assert obs.reward_breakdown.grounded_report >= 0.75


def test_structured_report_payload_can_complete_mission():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    obs = complete_capture_flow(env)
    photo_ids = [capture.photo_id for capture in obs.capture_log]
    thermal_ids = [capture.photo_id for capture in obs.capture_log if capture.sensor == "thermal"]
    defect_ids = [defect.defect_id for defect in env.debug_world.hidden_defects]

    obs = env.step(
        act(
            "submit_evidence_pack",
            mission_status="complete",
            evidence=[
                {
                    "requirement_id": "thermal_overview_rows_B4_B8",
                    "status": "satisfied",
                    "photo_ids": thermal_ids,
                }
            ],
            issues_found=[
                {"issue_id": defect_id, "evidence_photo_ids": photo_ids, "recommended_followup": "manual review"}
                for defect_id in defect_ids
            ],
            open_items=[],
            safety_notes=["Returned home with battery reserve."],
        )
    )

    assert obs.action_result["accepted"] is True
    assert obs.reward_breakdown.integrity_gate == 1.0
    assert obs.reward_breakdown.grounded_report >= 0.75


def test_false_positive_glare_is_not_required_issue_reward():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2203, scenario_family="false_positive_glare")

    score, debug = compute_issue_capture(env.debug_world)

    assert score == 1.0
    assert debug["issues_required"] == 0
    assert all(not defect.counts_for_issue_reward for defect in env.debug_world.hidden_defects)


def test_unknown_finding_id_is_integrity_penalized():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2202, scenario_family="bypass_diode_fault")
    world = env.debug_world
    known_defect = world.hidden_defects[0].defect_id
    world.final_report = EvidenceReport(findings=[{"finding": known_defect}, {"finding": "hallucinated_issue"}])

    integrity_gate, warnings = compute_integrity_gate(world)

    assert integrity_gate == 0.2
    assert any("unsupported issue claims" in warning for warning in warnings)


def test_natural_language_finding_is_not_treated_as_hallucinated_issue_id():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2202, scenario_family="bypass_diode_fault")
    world = env.debug_world
    world.final_report = EvidenceReport(findings=[{"finding": "Thermal anomaly observed on the inspected row"}])

    _, warnings = compute_integrity_gate(world)

    assert not any("unsupported issue claims" in warning for warning in warnings)


def test_process_reward_only_fires_on_progress_not_idle_actions():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=2101, scenario_family="single_hotspot")

    takeoff = env.step(act("takeoff", altitude_m=18))
    moved = env.step(act("move_to_asset", asset_id="row_B6", standoff_bucket="far", speed_mps=5))
    pointed = env.step(act("point_camera_at", asset_id="row_B6"))
    assert takeoff.reward_breakdown.process_reward == 0.0
    assert moved.reward_breakdown.process_reward == 0.0
    assert pointed.reward_breakdown.process_reward == 0.0

    env.step(act("set_camera_source", source="thermal"))
    captured = env.step(act("capture_thermal", label="thermal overview"))
    assert captured.reward_breakdown.process_reward > 0.0
