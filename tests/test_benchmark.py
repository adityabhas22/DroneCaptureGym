"""Smoke tests for the regression benchmark (issue #4)."""

from dronecaptureops.evaluation.benchmark import check_against_bands, run_benchmark


def test_benchmark_emits_summary_for_every_policy_bucket_combo():
    report = run_benchmark(
        policy_names=("scripted",),
        suite_names=("smoke",),
        task_ids=("basic_thermal_survey",),
        task_seeds=(2101,),
    )

    suite_summary = next(s for s in report.summaries if s.bucket == "suite" and s.bucket_name == "smoke")
    task_summary = next(s for s in report.summaries if s.bucket == "task" and s.bucket_name == "basic_thermal_survey")
    assert suite_summary.episodes > 0
    assert task_summary.episodes == 1
    assert "DroneCaptureOps Regression Benchmark" in report.to_markdown()


def test_benchmark_band_check_flags_out_of_band():
    report = run_benchmark(
        policy_names=("scripted",),
        suite_names=("smoke",),
        task_ids=(),
        task_seeds=(2101,),
    )
    in_band = check_against_bands(report, {"suite:smoke:scripted": {"mean_reward_min": -1.0, "mean_reward_max": 1.0}})
    assert in_band == []
    out_of_band = check_against_bands(report, {"suite:smoke:scripted": {"mean_reward_min": 1.5}})
    assert out_of_band  # mean reward can't be above 1.0
