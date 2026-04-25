"""Tests for the model evaluation harness (no vLLM needed)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.eval_models import (
    EvalCell,
    EvalRow,
    aggregate,
    build_plan,
    format_summary,
    resolve_tasks,
)
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


# --- task tiering ------------------------------------------------------------


def test_short_tier_excludes_multi_block():
    """Multi-block survey has max_steps=80 → it lands in `stretch`, not `short`."""

    short = resolve_tasks(tier="short", explicit=None)
    assert "multi_block_survey" not in short
    assert "basic_thermal_survey" in short


def test_stretch_tier_includes_multi_block():
    stretch = resolve_tasks(tier="stretch", explicit=None)
    assert "multi_block_survey" in stretch


def test_all_tier_returns_every_task():
    every = resolve_tasks(tier="all", explicit=None)
    assert set(every) == set(SOLAR_TASKS.keys())


def test_explicit_tasks_overrides_tier():
    chosen = resolve_tasks(tier="short", explicit=["multi_block_survey", "basic_thermal_survey"])
    assert chosen == ["multi_block_survey", "basic_thermal_survey"]


def test_explicit_tasks_with_unknown_id_raises():
    with pytest.raises(SystemExit):
        resolve_tasks(tier="short", explicit=["definitely_not_a_task"])


# --- plan layout -------------------------------------------------------------


def test_build_plan_cross_product_of_models_tasks_seeds():
    plan = build_plan(
        models=["m1", "m2"],
        task_ids=["basic_thermal_survey", "anomaly_confirmation"],
        seeds=3,
        seed_offset=10,
    )
    assert len(plan) == 2 * 2 * 3
    seeds = sorted({cell.seed for cell in plan})
    assert seeds == [10, 11, 12]
    assert {cell.model for cell in plan} == {"m1", "m2"}


def test_build_plan_pulls_max_steps_from_task_spec():
    plan = build_plan(models=["m"], task_ids=["limited_steps_rapid_survey"], seeds=1, seed_offset=0)
    assert plan[0].max_steps == SOLAR_TASKS["limited_steps_rapid_survey"].max_steps


# --- aggregation -------------------------------------------------------------


def _row(model: str, task_id: str, seed: int, *, reward: float, success: bool, steps: int, parse_errors: int = 0) -> EvalRow:
    return EvalRow(
        model=model,
        task_id=task_id,
        seed=seed,
        success=success,
        complete=success,
        total_reward=reward,
        steps=steps,
        parse_error_count=parse_errors,
    )


def test_aggregate_per_model_means():
    rows = [
        _row("m_a", "task_x", 0, reward=1.0, success=True, steps=20),
        _row("m_a", "task_x", 1, reward=0.5, success=False, steps=25),
        _row("m_b", "task_x", 0, reward=0.0, success=False, steps=10),
    ]
    summary = aggregate(rows)

    a = summary["per_model"]["m_a"]
    assert a["n"] == 2
    assert a["success_rate"] == pytest.approx(0.5)
    assert a["mean_reward"] == pytest.approx(0.75)

    b = summary["per_model"]["m_b"]
    assert b["n"] == 1
    assert b["success_rate"] == 0.0
    assert b["mean_reward"] == 0.0


def test_aggregate_per_cell_breaks_down_by_model_and_task():
    rows = [
        _row("m_a", "task_x", 0, reward=1.0, success=True, steps=20),
        _row("m_a", "task_y", 0, reward=0.2, success=False, steps=40),
    ]
    summary = aggregate(rows)
    cells = summary["per_cell"]
    assert "m_a::task_x" in cells
    assert "m_a::task_y" in cells
    assert cells["m_a::task_x"]["mean_reward"] == 1.0
    assert cells["m_a::task_y"]["success_rate"] == 0.0


def test_format_summary_runs_without_error():
    rows = [_row("m_a", "task_x", 0, reward=0.7, success=False, steps=20)]
    text = format_summary(aggregate(rows), rows)
    assert "per-model summary" in text
    assert "per-(model, task)" in text


# --- CLI dry-run -------------------------------------------------------------


def test_eval_models_dry_run_prints_plan(tmp_path: Path, capsys):
    """--dry-run should print the plan and exit 0 without loading vLLM."""

    import subprocess
    import sys as _sys

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            _sys.executable, "-m", "training.eval_models",
            "--models", "Qwen/Qwen3-4B-Instruct-2507",
            "--task-tier", "short",
            "--seeds", "1",
            "--output", str(tmp_path / "out.jsonl"),
            "--dry-run",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Qwen/Qwen3-4B-Instruct-2507" in proc.stdout
    assert "basic_thermal_survey" in proc.stdout
