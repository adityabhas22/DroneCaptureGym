"""CLI smoke tests for ``training/train_grpo.py``.

Heavy-lifting (model loading, GPU forwards) is gated behind ``--no-fit``
and the actual fit() call, so these tests run on CPU-only CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "training.train_grpo", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_works():
    result = _run_cli(["--help"])
    assert result.returncode == 0, result.stderr
    assert "GRPO trainer" in result.stdout


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
        "--prompts-per-step", "2",
        "--group-size", "3",
        "--actor-lr", "1e-5",
        "--kl-coef", "0.05",
        "--allow-fresh-lora",
    ])
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)["config"]
    assert cfg["total_steps"] == 5
    assert cfg["rollout"]["prompts_per_step"] == 2
    assert cfg["rollout"]["group_size"] == 3
    assert cfg["optimizer"]["actor_lr"] == pytest.approx(1e-5)
    assert cfg["algorithm"]["kl_coef"] == pytest.approx(0.05)
    assert cfg["allow_fresh_lora"] is True


def test_gpu_runtime_env_defaults(monkeypatch):
    from training.train_grpo import configure_gpu_runtime_env

    monkeypatch.delenv("PYTORCH_NVML_BASED_CUDA_CHECK", raising=False)
    configure_gpu_runtime_env()
    import os

    assert os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"


@pytest.mark.parametrize(
    ("config_path", "kl_coef", "expected_steps"),
    [
        ("training/configs/grpo_smoke_4b_l40.yaml", 0.01, 1),
        ("training/configs/grpo_tiny_4b_l40.yaml", 0.01, 10),
        ("training/configs/grpo_tiny_4b_l40_kl005.yaml", 0.005, 10),
        ("training/configs/grpo_tiny_4b_l40_kl02.yaml", 0.02, 10),
    ],
)
def test_grpo_configs_resolve(config_path, kl_coef, expected_steps):
    result = _run_cli(["--config", config_path, "--dry-run"])
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)["config"]
    assert cfg["model_name"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert cfg["allow_fresh_lora"] is False
    assert cfg["total_steps"] == expected_steps
    assert cfg["algorithm"]["kl_coef"] == pytest.approx(kl_coef)
    assert cfg["reporting"]["enabled"] is True


def test_grpo_smoke_config_is_minimal():
    result = _run_cli(["--config", "training/configs/grpo_smoke_4b_l40.yaml", "--dry-run"])
    assert result.returncode == 0, result.stderr
    cfg = json.loads(result.stdout)["config"]
    # Smoke config should be tiny: 1 prompt, 2 group samples, 1 step.
    assert cfg["rollout"]["prompts_per_step"] == 1
    assert cfg["rollout"]["group_size"] == 2
    assert cfg["total_steps"] == 1


def test_unknown_task_id_is_rejected(tmp_path: Path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "model_name: Qwen/Qwen3-4B-Instruct-2507\n"
        "allow_fresh_lora: true\n"
        "train_tasks:\n  - this_task_does_not_exist\n"
    )
    result = _run_cli(["--config", str(bad_yaml), "--dry-run"])
    assert result.returncode != 0
    assert "unknown task ids" in result.stderr.lower() or "unknown task ids" in result.stdout.lower()
