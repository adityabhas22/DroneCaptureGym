"""Tests for the SFT trainer's pure-Python pieces (no torch/transformers needed).

The full `train()` function depends on transformers/trl/peft; we don't run
it here. We test what we can: config loading, train/val split discipline,
and CLI override semantics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from training.sft_warmstart import (
    LoRAConfig,
    SFTTrainConfig,
    _apply_cli_overrides,
    load_config,
    load_jsonl,
    split_train_val,
)


# --- config ------------------------------------------------------------------


def test_load_config_returns_built_in_defaults_for_missing_path():
    config = load_config(Path("/tmp/this/path/does/not/exist.yaml"))
    assert isinstance(config, SFTTrainConfig)
    assert config.lora.enabled is True
    assert config.early_stopping_patience >= 1


def test_load_config_parses_yaml(tmp_path: Path):
    path = tmp_path / "trainer.yaml"
    path.write_text(
        "model_name: Qwen/Qwen2.5-7B-Instruct\n"
        "learning_rate: 5e-6\n"
        "num_train_epochs: 1\n"
        "lora:\n"
        "  enabled: false\n"
    )
    config = load_config(path)
    assert config.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert config.learning_rate == 5e-6
    assert config.num_train_epochs == 1
    assert config.lora.enabled is False


# --- CLI overrides -----------------------------------------------------------


def _ns(**kwargs) -> argparse.Namespace:
    """Build an argparse namespace with sensible defaults for unspecified flags."""

    defaults = {
        "dataset": None,
        "output_dir": None,
        "model_name": None,
        "num_train_epochs": None,
        "learning_rate": None,
        "max_seq_length": None,
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
        "early_stopping_patience": None,
        "seed": None,
        "no_lora": False,
        "no_eval": False,
        "dry_run": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_cli_override_disables_lora_and_eval():
    config = SFTTrainConfig()
    overridden = _apply_cli_overrides(config, _ns(no_lora=True, no_eval=True))
    assert overridden.lora.enabled is False
    assert overridden.val_seed_fraction == 0.0


def test_cli_override_propagates_lr_and_epochs():
    config = SFTTrainConfig()
    overridden = _apply_cli_overrides(config, _ns(num_train_epochs=1, learning_rate=1e-5))
    assert overridden.num_train_epochs == 1
    assert overridden.learning_rate == 1e-5


def test_cli_override_no_args_returns_same_config():
    config = SFTTrainConfig()
    assert _apply_cli_overrides(config, _ns()) == config


# --- train/val split ---------------------------------------------------------


def _records(task_id: str, seeds: list[int]) -> list[dict]:
    return [{"task_id": task_id, "seed": seed, "messages": [], "reward": 1.0} for seed in seeds]


def test_split_holds_out_seeds_within_each_task():
    records = (
        _records("task_a", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        + _records("task_b", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    )
    train, val = split_train_val(records, val_seed_fraction=0.2, seed=42)

    train_seeds = {(r["task_id"], r["seed"]) for r in train}
    val_seeds = {(r["task_id"], r["seed"]) for r in val}
    assert not (train_seeds & val_seeds), "train and val must not share (task, seed) pairs"

    # Each task gets at least one val seed.
    val_by_task = {task_id for task_id, _ in val_seeds}
    assert val_by_task == {"task_a", "task_b"}


def test_split_keeps_single_seed_tasks_in_train():
    """Tasks with only one seed can't be split — they go entirely to train."""

    records = _records("loner", [0]) + _records("normal", list(range(10)))
    train, val = split_train_val(records, val_seed_fraction=0.2, seed=42)

    train_loner = [r for r in train if r["task_id"] == "loner"]
    val_loner = [r for r in val if r["task_id"] == "loner"]
    assert len(train_loner) == 1
    assert len(val_loner) == 0


def test_split_is_deterministic_for_same_seed():
    records = _records("task_a", list(range(10)))
    train_a, val_a = split_train_val(records, val_seed_fraction=0.3, seed=11)
    train_b, val_b = split_train_val(records, val_seed_fraction=0.3, seed=11)

    train_seeds_a = sorted(r["seed"] for r in train_a)
    train_seeds_b = sorted(r["seed"] for r in train_b)
    assert train_seeds_a == train_seeds_b


def test_split_changes_with_different_seed():
    records = _records("task_a", list(range(10)))
    val_a = {r["seed"] for r in split_train_val(records, val_seed_fraction=0.3, seed=11)[1]}
    val_b = {r["seed"] for r in split_train_val(records, val_seed_fraction=0.3, seed=99)[1]}
    assert val_a != val_b, "different seeds should produce different val splits"


# --- jsonl loading -----------------------------------------------------------


def test_load_jsonl_round_trips(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps({"task_id": "x", "seed": 0, "messages": []}) + "\n"
        + json.dumps({"task_id": "y", "seed": 1, "messages": []}) + "\n"
    )
    records = load_jsonl(path)
    assert len(records) == 2
    assert records[0]["task_id"] == "x"


def test_load_jsonl_missing_file_raises():
    with pytest.raises(SystemExit):
        load_jsonl(Path("/tmp/definitely-missing.jsonl"))


# --- LoRA defaults -----------------------------------------------------------


def test_lora_defaults_target_qwen_friendly_modules():
    """LoRA defaults should hit attention + MLP projections so the warm-start
    has enough capacity to learn tool-call format without enabling full FT."""

    lora = LoRAConfig()
    assert lora.enabled is True
    assert "q_proj" in lora.target_modules
    assert "v_proj" in lora.target_modules
    assert "gate_proj" in lora.target_modules
    assert lora.r >= 8
