"""Tests for the RL trainer's pure-Python pieces (no torch/transformers needed).

Mirrors the discipline of `tests/test_sft_warmstart.py`: we exercise the
config loader, sampling plan, advantage computation, and CLI override
logic directly. The actual `train()` loop is exercised separately by
running the dry-run CLI in CI; full RL runs are out of scope for unit tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from training.train_grpo import (
    RLConfig,
    RolloutCell,
    _apply_cli_overrides,
    aggregate_iteration_stats,
    build_plan,
    compute_group_advantages,
    load_config,
)
from training.train_grpo import CellRolloutResult


# --- config ------------------------------------------------------------------


def test_load_config_returns_built_in_defaults_for_missing_path():
    config = load_config(Path("/tmp/this/path/does/not/exist.yaml"))
    assert isinstance(config, RLConfig)
    assert config.lora.enabled is True
    assert config.group_size >= 2


def test_load_config_parses_yaml(tmp_path: Path):
    path = tmp_path / "rl.yaml"
    path.write_text(
        "model_name: Qwen/Qwen3-4B-Instruct-2507\n"
        "num_iterations: 3\n"
        "group_size: 8\n"
        "kl_coef: 0.1\n"
        "train_tasks:\n"
        "  - basic_thermal_survey\n"
        "  - multi_anomaly_triage\n"
    )
    config = load_config(path)
    assert config.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.num_iterations == 3
    assert config.group_size == 8
    assert config.kl_coef == pytest.approx(0.1)
    assert config.train_tasks == ["basic_thermal_survey", "multi_anomaly_triage"]


def test_load_config_rejects_invalid_group_size(tmp_path: Path):
    path = tmp_path / "rl.yaml"
    path.write_text("group_size: 1\n")  # group_size must be >= 2 for relative advantages
    with pytest.raises(SystemExit):
        load_config(path)


# --- sampling plan -----------------------------------------------------------


def test_build_plan_is_deterministic_per_iteration():
    config = RLConfig(
        train_tasks=["alpha", "beta", "gamma"],
        tasks_per_iteration=3,
        seeds_per_task_iter=1,
        group_size=4,
        seed=123,
    )
    cells_a = build_plan(config, iteration=1)
    cells_b = build_plan(config, iteration=1)
    cells_other = build_plan(config, iteration=2)
    assert [(c.task_id, c.seed) for c in cells_a] == [(c.task_id, c.seed) for c in cells_b]
    assert [(c.task_id, c.seed) for c in cells_a] != [(c.task_id, c.seed) for c in cells_other]
    assert all(c.group_size == 4 for c in cells_a)


def test_build_plan_only_uses_train_tasks():
    config = RLConfig(
        train_tasks=["only_one_task"],
        tasks_per_iteration=5,
        seeds_per_task_iter=2,
        group_size=2,
    )
    cells = build_plan(config, iteration=1)
    assert len(cells) == 5 * 2
    assert {c.task_id for c in cells} == {"only_one_task"}


def test_build_plan_rejects_empty_train_tasks():
    config = RLConfig(train_tasks=[])
    with pytest.raises(SystemExit):
        build_plan(config, iteration=1)


# --- advantage computation ---------------------------------------------------


def test_compute_group_advantages_normalises():
    advantages = compute_group_advantages([0.0, 0.5, 1.0, 1.5], normalize=True)
    assert len(advantages) == 4
    # Symmetric inputs around their mean → mean(advantage) ~ 0.
    assert sum(advantages) == pytest.approx(0.0, abs=1e-6)
    # Highest reward -> highest advantage.
    assert advantages[-1] > advantages[0]


def test_compute_group_advantages_constant_rewards_returns_zero():
    advantages = compute_group_advantages([0.42, 0.42, 0.42], normalize=True)
    assert advantages == [0.0, 0.0, 0.0]


def test_compute_group_advantages_no_normalize_centres_only():
    advantages = compute_group_advantages([0.0, 1.0, 2.0], normalize=False)
    assert advantages == pytest.approx([-1.0, 0.0, 1.0])


def test_compute_group_advantages_handles_singleton():
    advantages = compute_group_advantages([0.7], normalize=True)
    assert advantages == [0.0]


def test_compute_group_advantages_empty_input():
    assert compute_group_advantages([]) == []


# --- aggregate stats ---------------------------------------------------------


def _result(reward: float, *, success: bool = True, parse_errors: int = 0) -> CellRolloutResult:
    return CellRolloutResult(
        cell=RolloutCell(task_id="basic_thermal_survey", seed=1, group_size=2),
        rollout_index=0,
        reward=reward,
        success=success,
        steps=10,
        parse_errors=parse_errors,
        safety_gate=1.0,
        integrity_gate=1.0,
        turn_count=10,
    )


def test_aggregate_iteration_stats_summarises_rewards():
    stats = aggregate_iteration_stats([_result(0.1), _result(0.5), _result(0.9)])
    assert stats["n"] == 3
    assert stats["mean_reward"] == pytest.approx(0.5)
    assert stats["max_reward"] == pytest.approx(0.9)
    assert stats["min_reward"] == pytest.approx(0.1)
    assert stats["success_rate"] == 1.0


def test_aggregate_iteration_stats_handles_empty():
    assert aggregate_iteration_stats([]) == {"n": 0}


def test_aggregate_iteration_stats_includes_parse_errors():
    stats = aggregate_iteration_stats([
        _result(0.0, parse_errors=1),
        _result(0.0, parse_errors=2),
    ])
    # 3 parse errors over 20 steps -> 0.15
    assert stats["parse_error_rate"] == pytest.approx(0.15, abs=1e-4)


# --- CLI overrides -----------------------------------------------------------


def _ns(**kwargs) -> argparse.Namespace:
    defaults = {
        "output_dir": None,
        "model_name": None,
        "sft_adapter": None,
        "num_iterations": None,
        "group_size": None,
        "seed": None,
        "no_lora": False,
        "no_kl": False,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_cli_override_disables_lora_and_kl():
    config = RLConfig()
    overridden = _apply_cli_overrides(config, _ns(no_lora=True, no_kl=True))
    assert overridden.lora.enabled is False
    assert overridden.kl_coef == 0.0


def test_cli_override_sets_iterations_and_group_size():
    config = RLConfig()
    overridden = _apply_cli_overrides(config, _ns(num_iterations=2, group_size=8))
    assert overridden.num_iterations == 2
    assert overridden.group_size == 8


def test_cli_override_records_sft_adapter():
    config = RLConfig()
    overridden = _apply_cli_overrides(config, _ns(sft_adapter="/tmp/adapter"))
    assert overridden.sft_adapter_path == "/tmp/adapter"


def test_cli_override_no_args_returns_same_config():
    config = RLConfig()
    overridden = _apply_cli_overrides(config, _ns())
    assert overridden is config
