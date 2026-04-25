"""Smoke tests for the inference CLI and offline policy adapters."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "inference.py", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_inference_cli_scripted_basic_thermal_survey():
    rc, stdout, stderr = _run("--task", "basic_thermal_survey", "--policy", "scripted")
    assert rc == 0, stderr
    summary = json.loads(stdout)
    assert summary["task_id"] == "basic_thermal_survey"
    assert summary["policy"] == "scripted"
    assert summary["success"] is True
    assert summary["complete"] is True
    assert summary["total_reward"] >= 0.95


def test_inference_cli_random_runs_without_crash():
    rc, stdout, _ = _run("--task", "anomaly_confirmation", "--policy", "random", "--max-steps", "4", "--seed", "11")
    assert rc == 0
    summary = json.loads(stdout)
    assert summary["policy"] == "random"
    assert summary["steps"] <= 4


def test_inference_cli_writes_trajectory_and_messages_to_disk(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    messages_path = tmp_path / "messages.jsonl"
    rc, _, stderr = _run(
        "--task",
        "anomaly_confirmation",
        "--policy",
        "scripted",
        "--output",
        str(trace_path),
        "--messages-output",
        str(messages_path),
    )
    assert rc == 0, stderr
    assert trace_path.exists()
    assert messages_path.exists()

    record = json.loads(messages_path.read_text())
    assert record["task_id"] == "anomaly_confirmation"
    messages = record["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert any(message["role"] == "assistant" for message in messages)
    # System message should include the dynamic tool catalog.
    assert "capture_thermal" in messages[0]["content"]
    assert "anomaly_confirmation" in messages[0]["content"]
