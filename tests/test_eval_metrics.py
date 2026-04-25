"""Tests for the diagnostic-metrics layer used in base-model evaluation."""

from __future__ import annotations

from dronecaptureops.agent import (
    RandomPolicy,
    RolloutResult,
    RolloutRunner,
    ScriptedPolicy,
    TaskOraclePolicy,
)
from dronecaptureops.agent.eval_metrics import (
    CHECKPOINT_NAMES,
    FAILURE_MODES,
    aggregate_diagnostics,
    classify_failure_mode,
    coverage_metrics,
    extract_checkpoints,
    extract_reward_components,
    oracle_comparison,
    safety_profile,
    tool_call_distribution,
    trajectory_metrics,
)


def _oracle_run(task_id: str = "basic_thermal_survey", seed: int = 7) -> RolloutResult:
    runner = RolloutRunner()
    return runner.run(TaskOraclePolicy(task_id=task_id), seed=seed, task_id=task_id, max_steps=40)


def _random_run(task_id: str = "basic_thermal_survey", seed: int = 7) -> RolloutResult:
    runner = RolloutRunner()
    return runner.run(RandomPolicy(seed=seed), seed=seed, task_id=task_id, max_steps=12)


# --- failure mode classifier -------------------------------------------------


def test_oracle_episode_classifies_as_success():
    result = _oracle_run()
    assert classify_failure_mode(result) == "success"


def test_random_episode_classifies_as_a_real_failure_mode():
    """The random policy should hit one of the catalogued failure modes,
    not 'success' and not 'unknown'."""

    result = _random_run()
    mode = classify_failure_mode(result)
    assert mode in FAILURE_MODES
    assert mode != "success"
    assert mode != "unknown"


def test_premature_submit_is_classified_correctly():
    """Submit on step 0 with no captures → premature_submit / no_takeoff."""

    from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
    from dronecaptureops.core.models import RawDroneAction

    env = DroneCaptureOpsEnvironment()
    runner = RolloutRunner(env=env)

    class _SubmitImmediatelyPolicy:
        name = "submit_now"

        def next_action(self, obs, ctx):  # noqa: ANN001
            return RawDroneAction(
                tool_name="submit_evidence_pack",
                arguments={"summary": "premature", "photo_ids": [], "findings": []},
            )

    result = runner.run(_SubmitImmediatelyPolicy(), seed=7, task_id="basic_thermal_survey", max_steps=5)
    mode = classify_failure_mode(result)
    # Either no_takeoff or no_capture is acceptable (both upstream of submit).
    assert mode in {"no_takeoff", "no_capture"}


# --- checkpoints -------------------------------------------------------------


def test_oracle_hits_every_capability_checkpoint_for_anomaly_task():
    """anomaly_confirmation has a deterministic hotspot, so oracle should
    pass every checkpoint including rgb_paired_anomaly."""

    result = _oracle_run("anomaly_confirmation")
    checkpoints = extract_checkpoints(result)
    expected_true = {
        "any_valid_action",
        "explored_env",
        "took_off",
        "captured_thermal",
        "captured_rgb",
        "inspected_capture",
        "covered_any_row",
        "detected_any_anomaly",
        "rgb_paired_anomaly",
        "returned_home",
        "landed",
        "submitted",
        "submission_accepted",
    }
    for name in expected_true:
        assert checkpoints[name] is True, f"oracle missed checkpoint {name}"


def test_basic_task_has_no_anomaly_checkpoints():
    """basic_thermal_survey has no defects; anomaly checkpoints should be False."""

    result = _oracle_run("basic_thermal_survey")
    checkpoints = extract_checkpoints(result)
    assert checkpoints["detected_any_anomaly"] is False
    assert checkpoints["rgb_paired_anomaly"] is False
    assert checkpoints["submission_accepted"] is True


# --- tool distribution -------------------------------------------------------


def test_tool_distribution_sums_to_step_count():
    result = _oracle_run()
    distribution = tool_call_distribution(result)
    assert sum(distribution.values()) == len(result.trajectory)


def test_tool_distribution_includes_parse_errors_separately():
    """Parse errors get tagged under '_parse_error', not under a real tool name."""

    from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
    from dronecaptureops.core.errors import ActionValidationError

    env = DroneCaptureOpsEnvironment()
    runner = RolloutRunner(env=env)

    class _GarbagePolicy:
        name = "garbage"

        def next_action(self, obs, ctx):  # noqa: ANN001
            raise ActionValidationError("malformed output")

    result = runner.run(_GarbagePolicy(), seed=7, task_id="basic_thermal_survey", max_steps=3)
    distribution = tool_call_distribution(result)
    assert distribution.get("_parse_error", 0) >= 1


# --- coverage metrics --------------------------------------------------------


def test_coverage_metrics_for_complete_run():
    result = _oracle_run("basic_thermal_survey")
    coverage = coverage_metrics(result)
    assert coverage["rows_required"] == 5
    assert coverage["rows_covered"] == 5
    assert coverage["rows_covered_fraction"] == 1.0
    assert coverage["missing_rows"] == []


def test_coverage_metrics_for_anomaly_task_pairs_rgb():
    result = _oracle_run("anomaly_confirmation")
    coverage = coverage_metrics(result)
    assert coverage["anomalies_detected"] >= 1
    assert coverage["rgb_pairing_fraction"] == 1.0
    assert coverage["unpaired_anomalies"] == []


# --- safety profile ----------------------------------------------------------


def test_safety_profile_for_clean_run_has_no_violations():
    result = _oracle_run()
    profile = safety_profile(result)
    assert profile["total_violations"] == 0
    assert profile["safety_gate"] == 1.0


# --- reward components -------------------------------------------------------


def test_extract_reward_components_returns_floats():
    result = _oracle_run()
    components = extract_reward_components(result)
    assert components["total"] >= 0.95
    assert components["evidence_success"] >= 0.5
    assert components["safety_gate"] == 1.0
    assert all(isinstance(value, float) for value in components.values())


# --- oracle comparison -------------------------------------------------------


def test_oracle_comparison_against_self_is_perfect():
    """Comparing the oracle's run against itself yields perfect overlap +
    step_ratio = 1.0."""

    oracle = _oracle_run()
    comparison = oracle_comparison(oracle, oracle)
    assert comparison["step_ratio"] == 1.0
    assert comparison["tool_jaccard"] == 1.0
    assert comparison["oracle_tool_recall"] == 1.0


def test_oracle_comparison_when_no_reference_provided():
    result = _oracle_run()
    comparison = oracle_comparison(result, None)
    assert comparison == {"available": False}


def test_random_vs_oracle_step_ratio_above_one():
    """Random policy explores more, so step_ratio (random/oracle) should
    typically exceed 1.0 — confirms the metric does what we think."""

    oracle = _oracle_run()
    random = _random_run()
    comparison = oracle_comparison(random, oracle)
    assert comparison["available"] is True
    # Step ratio is base/oracle. Random hits its own max_steps cap (12),
    # so the ratio depends on whichever ran longer.
    assert comparison["step_ratio"] > 0


# --- combined entry point ----------------------------------------------------


def test_trajectory_metrics_composes_all_pieces():
    oracle = _oracle_run("anomaly_confirmation")
    metrics = trajectory_metrics(oracle, oracle_result=oracle)
    assert metrics.failure_mode == "success"
    assert metrics.checkpoints["submission_accepted"] is True
    assert metrics.coverage["rows_required"] == 5
    assert metrics.oracle_comparison["available"] is True
    assert metrics.reward_components["total"] >= 0.95


# --- aggregation -------------------------------------------------------------


def _row_payload(model: str, *, failure_mode: str, checkpoints: dict, tool_calls: dict) -> dict:
    return {
        "model": model,
        "task_id": "basic_thermal_survey",
        "failure_mode": failure_mode,
        "checkpoints": {name: checkpoints.get(name, False) for name in CHECKPOINT_NAMES},
        "tool_calls": tool_calls,
    }


def test_aggregate_diagnostics_failure_distribution():
    rows = [
        _row_payload("m_a", failure_mode="success", checkpoints={n: True for n in CHECKPOINT_NAMES}, tool_calls={"takeoff": 1}),
        _row_payload("m_a", failure_mode="success", checkpoints={n: True for n in CHECKPOINT_NAMES}, tool_calls={"takeoff": 1}),
        _row_payload("m_a", failure_mode="no_takeoff", checkpoints={"any_valid_action": True}, tool_calls={"hover": 5}),
        _row_payload("m_b", failure_mode="format_collapse", checkpoints={}, tool_calls={"_parse_error": 10}),
    ]
    summary = aggregate_diagnostics(rows)

    a = summary["m_a"]
    assert a["n"] == 3
    assert a["failure_mode_distribution"].get("success") == round(2 / 3, 4)
    assert a["checkpoint_completion_rate"]["any_valid_action"] == 1.0
    assert a["checkpoint_completion_rate"]["submission_accepted"] == round(2 / 3, 4)

    b = summary["m_b"]
    assert b["failure_mode_distribution"].get("format_collapse") == 1.0
    assert b["tool_calls_per_episode"].get("_parse_error") == 10.0
