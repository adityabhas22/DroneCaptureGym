import json

from dronecaptureops.evaluation.policies import get_policy
from dronecaptureops.evaluation.rollout import RolloutRunner
from dronecaptureops.evaluation.suite_runner import run_suite
from dronecaptureops.evaluation.tracing import build_trace_artifacts, trace_rollout, write_trace_artifacts


def test_rollout_runner_captures_full_trajectory_and_reward_breakdowns():
    rollout = RolloutRunner().run(
        get_policy("scripted"),
        seed=2101,
        scenario_family="single_hotspot",
        max_steps=30,
    )

    assert rollout.trajectory
    assert rollout.final_observation["done"] is True
    assert "total" in rollout.reward_breakdown
    assert any(step.reward_breakdown for step in rollout.trajectory)
    assert any("total" in step.reward_breakdown for step in rollout.trajectory)
    assert any(step.reward_delta for step in rollout.trajectory)


def test_trace_artifacts_include_rewards_route_evidence_and_report(tmp_path):
    rollout = RolloutRunner().run(
        get_policy("scripted"),
        seed=2101,
        scenario_family="single_hotspot",
        max_steps=30,
    )
    artifacts = build_trace_artifacts(rollout)

    assert artifacts["episode_steps"]
    assert artifacts["route_log"]
    assert artifacts["evidence_log"]
    assert artifacts["inspection_report"]["submitted"] is True
    assert artifacts["trace"]["steps"][0]["reward_breakdown"]
    assert "Reward Deltas" in artifacts["trace_markdown"]

    paths = write_trace_artifacts(rollout, tmp_path)
    assert json.loads(paths["episode_steps"].read_text())
    assert json.loads(paths["trace"].read_text())["final_reward_breakdown"]
    assert paths["trace_markdown"].read_text().startswith("# DroneCaptureOps Trace")


def test_suite_runner_aggregates_dynamic_reward_columns():
    report = run_suite(get_policy("weak_scripted"), suite="smoke")

    assert report.rows
    assert report.episodes == 3
    assert "total" in report.reward_breakdown_mean
    assert all(row.reward_delta_totals for row in report.rows)
    assert "Reward Columns" in report.to_markdown()


def test_trace_artifacts_include_diagnostic_layers():
    """The v2 trace artifacts surface inspection diagnostics through the
    `evidence_log`, `inspection_report`, `route_log`, and `trace` payloads.

    The legacy `capture_table` / `reward_evolution` / `safety_timeline` /
    `final_report_diagnostics` keys were removed — this test now asserts the
    successor surface exposes equivalent debugging signal.
    """

    rollout = RolloutRunner().run(
        get_policy("scripted"),
        seed=2101,
        scenario_family="single_hotspot",
        max_steps=30,
    )
    artifacts = build_trace_artifacts(rollout)

    # Evidence log is the successor of capture_table.
    assert artifacts["evidence_log"], "evidence_log missing"
    first_capture = artifacts["evidence_log"][0]
    assert {"photo_id", "sensor"} <= set(first_capture)

    # Per-step reward breakdowns inside the trace replace reward_evolution.
    trace_steps = artifacts["trace"]["steps"]
    assert trace_steps, "trace steps missing"
    assert any("total" in step.get("reward_breakdown", {}) for step in trace_steps)

    # Route log records every step (proxy for safety_timeline/route signal).
    assert isinstance(artifacts["route_log"], list)

    # Inspection report subsumes final_report_diagnostics.
    inspection = artifacts["inspection_report"]
    assert "submitted" in inspection
    assert "checklist_status" in inspection


def test_trace_state_changes_show_policy_steps():
    rollout = RolloutRunner().run(
        get_policy("scripted"),
        seed=2101,
        scenario_family="single_hotspot",
        max_steps=30,
    )
    trace = trace_rollout(rollout)

    changed_paths = {
        change.path
        for step in trace.steps
        for change in step.state_changes
    }
    assert "telemetry.autopilot.mode" in changed_paths
    assert "evidence_artifacts" in changed_paths
