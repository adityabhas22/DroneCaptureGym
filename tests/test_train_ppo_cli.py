"""CLI smoke tests for training/train_ppo.py.

Heavy-lifting (model loading, vLLM, GPU) is gated behind --no-vllm and
the actual fit() call, so these tests can run on CPU-only CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "training.train_ppo", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_works():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "PPO trainer" in result.stdout


def test_dry_run_prints_resolved_config():
    result = _run_cli(["--dry-run"])
    assert result.returncode == 0, f"dry-run failed: stderr={result.stderr}"
    payload = json.loads(result.stdout)
    assert "config" in payload
    assert "n_train_tasks" in payload
    assert payload["n_train_tasks"] > 0
    assert "held_out_tasks" in payload


def test_dry_run_respects_cli_overrides():
    result = _run_cli([
        "--dry-run",
        "--total-steps", "5",
        "--minibatch-size", "8",
        "--actor-lr", "5e-6",
        "--kl-coef", "0.02",
        "--allow-fresh-lora",
    ])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    cfg = payload["config"]
    assert cfg["total_steps"] == 5
    assert cfg["minibatch_size"] == 8
    assert cfg["optimizer"]["actor_lr"] == pytest.approx(5e-6)
    assert cfg["algorithm"]["kl_coef"] == pytest.approx(0.02)
    assert cfg["allow_fresh_lora"] is True


def test_gpu_runtime_env_defaults_are_set_before_heavy_imports(monkeypatch):
    from training.train_ppo import configure_gpu_runtime_env

    monkeypatch.delenv("PYTORCH_NVML_BASED_CUDA_CHECK", raising=False)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)
    configure_gpu_runtime_env()
    assert __import__("os").environ["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
    assert __import__("os").environ["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"


def test_dry_run_held_out_tasks_excluded_from_train():
    result = _run_cli(["--dry-run"])
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    held_out = set(payload["held_out_tasks"])
    train_sample = set(payload["train_tasks_sample"])
    assert not (train_sample & held_out), (
        f"train tasks contain held-out: {train_sample & held_out}"
    )


@pytest.mark.parametrize(
    ("config_path", "expected_model", "allow_fresh_lora"),
    [
        ("training/configs/ppo_smoke_1p7b_l4.yaml", "Qwen/Qwen3-1.7B", True),
        ("training/configs/ppo_smoke_4b_a100.yaml", "Qwen/Qwen3-4B-Instruct-2507", False),
    ],
)
def test_smoke_configs_are_tiny_dry_runs(config_path, expected_model, allow_fresh_lora):
    result = _run_cli(["--config", config_path, "--dry-run"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    cfg = payload["config"]
    assert cfg["model_name"] == expected_model
    assert cfg["allow_fresh_lora"] is allow_fresh_lora
    assert cfg["total_steps"] == 1
    assert cfg["rollout"]["rollout_batch_size"] == 1
    assert payload["n_train_tasks"] == 1


def test_tiny_learning_config_is_multi_step_but_bounded():
    result = _run_cli(["--config", "training/configs/ppo_tiny_learning_4b.yaml", "--dry-run"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    cfg = payload["config"]
    assert cfg["model_name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg["allow_fresh_lora"] is False
    assert cfg["total_steps"] == 10
    assert cfg["rollout"]["rollout_batch_size"] == 4
    assert cfg["reporting"]["enabled"] is True
    assert payload["n_train_tasks"] == 4


@pytest.mark.parametrize(
    ("config_path", "kl_coef"),
    [
        ("training/configs/ppo_tiny_learning_4b_kl005.yaml", 0.005),
        ("training/configs/ppo_tiny_learning_4b_kl02.yaml", 0.02),
    ],
)
def test_tiny_learning_kl_branch_configs(config_path, kl_coef):
    result = _run_cli(["--config", config_path, "--dry-run"])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    cfg = payload["config"]
    assert cfg["total_steps"] == 10
    assert cfg["algorithm"]["kl_coef"] == pytest.approx(kl_coef)
    assert cfg["reporting"]["enabled"] is True


def test_eval_policy_held_out_matches_ppo_config():
    from training.eval_policy import DEFAULT_HELD_OUT

    config = yaml.safe_load((REPO_ROOT / "training/configs/ppo_train_default.yaml").read_text())
    assert DEFAULT_HELD_OUT == config["eval"]["held_out_tasks"]
