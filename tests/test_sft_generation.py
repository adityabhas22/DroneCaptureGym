"""Tests for the dynamic SFT data generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.generate_sft_data import (
    PolicyEntry,
    SFTGenConfig,
    TaskOverride,
    generate,
    resolve_task_plans,
)
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


def _basic_config(**overrides) -> SFTGenConfig:
    """Build a tiny config that runs in seconds."""

    base = {
        "seeds_per_task": 1,
        "max_steps": 30,
        "policies": [PolicyEntry(name="task_oracle")],
        "deduplicate": True,
        "shuffle": False,
        "held_out_tasks": ["multi_anomaly_triage"],
    }
    base.update(overrides)
    return SFTGenConfig(**base)


# --- plan resolution ---------------------------------------------------------


def test_resolve_task_plans_pulls_in_every_task_by_default():
    """include_tasks=None ⇒ every task in SOLAR_TASKS makes the cut.

    This is the contract that lets new tasks plug in without editing the
    generator: as soon as `solar_tasks.SOLAR_TASKS` grows, plan resolution
    grows.
    """

    plans = resolve_task_plans(_basic_config())
    plan_ids = {plan.task_id for plan in plans}
    assert plan_ids == set(SOLAR_TASKS.keys())


def test_resolve_task_plans_marks_held_out_correctly():
    plans = resolve_task_plans(_basic_config())
    held_out = {plan.task_id for plan in plans if plan.held_out}
    assert held_out == {"multi_anomaly_triage"}


def test_resolve_task_plans_respects_exclude():
    config = _basic_config(exclude_tasks=["limited_steps_rapid_survey", "no_anomaly_clearance"])
    plans = resolve_task_plans(config)
    plan_ids = {plan.task_id for plan in plans}
    assert "limited_steps_rapid_survey" not in plan_ids
    assert "no_anomaly_clearance" not in plan_ids
    assert "basic_thermal_survey" in plan_ids


def test_resolve_task_plans_respects_include_subset():
    config = _basic_config(include_tasks=["basic_thermal_survey"])
    plans = resolve_task_plans(config)
    assert {plan.task_id for plan in plans} == {"basic_thermal_survey"}


def test_resolve_task_plans_applies_per_task_overrides():
    config = _basic_config(
        tasks=[
            TaskOverride(task_id="basic_thermal_survey", seeds=[42, 43, 44], max_steps=10),
            TaskOverride(task_id="anomaly_confirmation", seeds_per_task=3),
        ]
    )
    plans_by_id = {plan.task_id: plan for plan in resolve_task_plans(config)}
    basic = plans_by_id["basic_thermal_survey"]
    assert basic.seeds == [42, 43, 44]
    assert basic.max_steps == 10
    anomaly = plans_by_id["anomaly_confirmation"]
    assert len(anomaly.seeds) == 3


def test_resolve_task_plans_skips_unknown_task_in_include(caplog):
    config = _basic_config(include_tasks=["basic_thermal_survey", "definitely_not_a_real_task"])
    plans = resolve_task_plans(config)
    assert {plan.task_id for plan in plans} == {"basic_thermal_survey"}


# --- end-to-end generation ---------------------------------------------------


@pytest.fixture
def tmp_outputs(tmp_path: Path):
    return {
        "train": tmp_path / "train.jsonl",
        "heldout": tmp_path / "heldout.jsonl",
    }


def test_generate_produces_chat_format_records(tmp_outputs):
    config = _basic_config(
        include_tasks=["basic_thermal_survey", "anomaly_confirmation"],
        held_out_tasks=[],
    )
    stats = generate(config, output_path=tmp_outputs["train"], heldout_output_path=None)

    assert stats["basic_thermal_survey"].rollouts_kept == 1
    assert stats["anomaly_confirmation"].rollouts_kept == 1
    assert tmp_outputs["train"].exists()

    records = [json.loads(line) for line in tmp_outputs["train"].read_text().splitlines()]
    assert len(records) == 2
    for record in records:
        assert "messages" in record
        assert record["messages"][0]["role"] == "system"
        assert record["messages"][1]["role"] == "user"
        assert any(m["role"] == "assistant" for m in record["messages"])
        assert record["success"] is True
        assert record["reward"] >= 0.95


def test_generate_segregates_held_out_tasks(tmp_outputs):
    config = _basic_config(
        include_tasks=["basic_thermal_survey", "multi_anomaly_triage"],
        held_out_tasks=["multi_anomaly_triage"],
    )
    generate(
        config,
        output_path=tmp_outputs["train"],
        heldout_output_path=tmp_outputs["heldout"],
        with_heldout=True,
    )

    train_records = [json.loads(line) for line in tmp_outputs["train"].read_text().splitlines()]
    heldout_records = [json.loads(line) for line in tmp_outputs["heldout"].read_text().splitlines()]

    assert {r["task_id"] for r in train_records} == {"basic_thermal_survey"}
    assert {r["task_id"] for r in heldout_records} == {"multi_anomaly_triage"}
    assert all(r["held_out"] is False for r in train_records)
    assert all(r["held_out"] is True for r in heldout_records)


def test_generate_respects_require_success(tmp_outputs):
    """Random policy fails everything; with require_success=True nothing is written."""

    config = _basic_config(
        include_tasks=["basic_thermal_survey"],
        held_out_tasks=[],
        policies=[PolicyEntry(name="random")],
        seeds_per_task=2,
        max_steps=5,
        require_success=True,
    )
    stats = generate(config, output_path=tmp_outputs["train"])
    assert stats["basic_thermal_survey"].rollouts_kept == 0
    assert stats["basic_thermal_survey"].rollouts_failed_success == 2
    assert tmp_outputs["train"].read_text() == ""


def test_generate_deduplicates_identical_trajectories(tmp_outputs):
    """task_oracle is deterministic per (task_id, seed); two replicates of the
    same seed produce identical messages, so deduplicate=True keeps only one.
    """

    config = _basic_config(
        include_tasks=["basic_thermal_survey"],
        held_out_tasks=[],
        policies=[PolicyEntry(name="task_oracle", weight=3)],
        seeds_per_task=1,
        deduplicate=True,
    )
    stats = generate(config, output_path=tmp_outputs["train"])
    assert stats["basic_thermal_survey"].rollouts_attempted == 3
    assert stats["basic_thermal_survey"].rollouts_kept == 1
    assert stats["basic_thermal_survey"].rollouts_dropped_duplicate == 2


def test_generate_writes_messages_in_tool_calls_dialect_when_requested(tmp_outputs):
    config = _basic_config(
        include_tasks=["basic_thermal_survey"],
        held_out_tasks=[],
        use_tool_calls=True,
    )
    generate(config, output_path=tmp_outputs["train"])
    record = json.loads(tmp_outputs["train"].read_text().splitlines()[0])
    assistants = [m for m in record["messages"] if m["role"] == "assistant"]
    assert assistants
    assert "tool_calls" in assistants[0]
    assert assistants[0]["tool_calls"][0]["type"] == "function"
