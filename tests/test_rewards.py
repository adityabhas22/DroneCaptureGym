from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import RawDroneAction


def act(tool_name: str, **arguments):
    return RawDroneAction(tool_name=tool_name, arguments=arguments)


def test_reward_breakdown_has_expected_shape():
    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=7)

    keys = set(obs.reward_breakdown.model_dump())

    assert {
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
