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
    ])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    cfg = payload["config"]
    assert cfg["total_steps"] == 5
    assert cfg["minibatch_size"] == 8
    assert cfg["optimizer"]["actor_lr"] == pytest.approx(5e-6)
    assert cfg["algorithm"]["kl_coef"] == pytest.approx(0.02)


def test_dry_run_held_out_tasks_excluded_from_train():
    result = _run_cli(["--dry-run"])
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    held_out = set(payload["held_out_tasks"])
    train_sample = set(payload["train_tasks_sample"])
    assert not (train_sample & held_out), (
        f"train tasks contain held-out: {train_sample & held_out}"
    )
