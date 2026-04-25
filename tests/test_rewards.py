from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def capture_thermal_overview(env: DroneCaptureOpsEnvironment):
    env.step(act("takeoff", altitude_m=18))
    env.step(act("fly_to_viewpoint", x=0, y=16, z=18, yaw_deg=0, speed_mps=5))
    env.step(act("fly_to_viewpoint", x=30, y=24, z=18, yaw_deg=-90, speed_mps=5))
    env.step(act("set_gimbal", pitch_deg=-60, yaw_deg=0))
    return env.step(act("capture_thermal", label="overview"))


def complete_capture_flow(env: DroneCaptureOpsEnvironment):
    obs = capture_thermal_overview(env)
    env.step(act("inspect_capture", photo_id=obs.last_capture.photo_id))
    env.step(act("fly_to_viewpoint", x=30, y=16, z=12, yaw_deg=-90, speed_mps=4))
    obs = env.step(act("capture_rgb", label="rgb anomaly context"))
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
    assert obs.reward_breakdown.total <= 0.2
    assert obs.reward_breakdown.debug["terminal_submitted"] is False
    assert obs.reward_breakdown.debug["nonterminal_cap_applied"] is True
    assert obs.reward_breakdown.debug["shaping_reward"] == obs.reward_breakdown.total
    assert obs.reward_breakdown.debug["raw_outcome_if_submitted"] > obs.reward_breakdown.total


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


def test_useful_rgb_redundancy_is_not_penalized():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    capture_thermal_overview(env)
    env.step(act("fly_to_viewpoint", x=30, y=16, z=12, yaw_deg=-90, speed_mps=4))

    obs = env.step(act("capture_rgb", label="rgb anomaly context"))

    assert obs.reward_breakdown.penalties == 0.0
    assert obs.reward_breakdown.issue_capture == 1.0


def test_repeated_harmless_calls_do_not_farm_process_reward():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = None
    for _ in range(20):
        obs = env.step(act("get_telemetry"))

    assert obs is not None
    assert obs.reward_breakdown.process_reward == 0.0
    assert obs.reward_breakdown.total == 0.0


def test_early_submission_cannot_score_highly():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)

    obs = env.step(act("submit_evidence_pack", summary="done", photo_ids=[], findings=[]))

    assert obs.reward_breakdown.evidence_success < 0.5
    assert obs.reward_breakdown.total <= 0.2


def test_thermal_only_report_cannot_fully_satisfy_rgb_required_anomalies():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    obs = capture_thermal_overview(env)
    photo_ids = [capture.photo_id for capture in obs.capture_log]
    defect_ids = [defect.defect_id for defect in env.debug_world.hidden_defects]

    obs = env.step(
        act(
            "submit_evidence_pack",
            summary=f"Rows inspected. Issues: {' '.join(defect_ids)}.",
            photo_ids=photo_ids,
            findings=[{"finding": defect_id, "photo_ids": photo_ids} for defect_id in defect_ids],
            safety_notes=["Inspection incomplete."],
        )
    )

    assert obs.action_result["accepted"] is False
    assert obs.reward_breakdown.debug["cited_issue_capture"] == 0.6
    assert obs.reward_breakdown.total <= 0.6


def test_claimed_return_home_note_is_capped_when_telemetry_disagrees():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    obs = capture_thermal_overview(env)
    photo_ids = [capture.photo_id for capture in obs.capture_log]

    obs = env.step(
        act(
            "submit_evidence_pack",
            summary="Rows inspected. Returned home with battery reserve.",
            photo_ids=photo_ids,
            findings=[],
            safety_notes=["Returned home with battery reserve."],
        )
    )

    assert obs.action_result["accepted"] is False
    assert obs.reward_breakdown.integrity_gate <= 0.4
    assert obs.reward_breakdown.total <= 0.4


def test_missing_cited_thermal_coverage_lowers_terminal_success():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    obs = complete_capture_flow(env)
    rgb_ids = [capture.photo_id for capture in obs.capture_log if capture.sensor == "rgb"]

    obs = env.step(
        act(
            "submit_evidence_pack",
            mission_status="complete",
            evidence=[
                {
                    "requirement_id": "thermal_overview_rows_B4_B8",
                    "status": "satisfied",
                    "photo_ids": rgb_ids,
                }
            ],
            issues_found=[],
            open_items=[],
            safety_notes=["Returned home with battery reserve."],
        )
    )

    assert obs.action_result["accepted"] is False
    assert obs.reward_breakdown.debug["cited_required_coverage"] == 0.0
    assert obs.reward_breakdown.evidence_success < 0.5


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
    assert obs.reward_breakdown.debug["cited_required_coverage"] == 1.0
    assert obs.reward_breakdown.grounded_report >= 0.75
