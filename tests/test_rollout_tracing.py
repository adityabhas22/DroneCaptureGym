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
    """Issue #5: capture_table, reward_evolution, safety_timeline, final_report_diagnostics."""

    rollout = RolloutRunner().run(
        get_policy("scripted"),
        seed=2101,
        scenario_family="single_hotspot",
        max_steps=30,
    )
    artifacts = build_trace_artifacts(rollout)

    assert artifacts["capture_table"], "capture_table missing"
    first_capture = artifacts["capture_table"][0]
    assert {"photo_id", "sensor", "targets_visible", "quality_score", "cited_in_report"} <= set(first_capture)

    assert artifacts["reward_evolution"], "reward_evolution missing"
    first_reward = artifacts["reward_evolution"][0]
    assert "components" in first_reward and "total" in first_reward["components"]

    # Safety timeline can be empty when no violations occurred — list type still required.
    assert isinstance(artifacts["safety_timeline"], list)

    diagnostics = artifacts["final_report_diagnostics"]
    assert {
        "submitted",
        "missing_cited_rows",
        "fake_photo_ids_cited",
        "low_quality_cited_photo_ids",
        "integrity_warnings",
    } <= set(diagnostics)


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
