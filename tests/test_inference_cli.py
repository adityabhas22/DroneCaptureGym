"""Smoke tests for the inference CLI and offline policy adapters."""

from __future__ import annotations

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
    lines = stdout.splitlines()
    assert lines[0] == "[START] task=basic_thermal_survey env=dronecaptureops-gym model=scripted"
    assert any(line.startswith("[STEP] step=1 action=") for line in lines)
    assert lines[-1].startswith("[END] success=true ")
    assert "score=1.00" in lines[-1]


def test_inference_cli_random_runs_without_crash():
    rc, stdout, _ = _run("--task", "anomaly_confirmation", "--policy", "random", "--max-steps", "4", "--seed", "11")
    assert rc == 0
    lines = stdout.splitlines()
    assert lines[0] == "[START] task=anomaly_confirmation env=dronecaptureops-gym model=random"
    step_lines = [line for line in lines if line.startswith("[STEP]")]
    assert 1 <= len(step_lines) <= 4
    assert lines[-1].startswith("[END] ")


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

    import json

    record = json.loads(messages_path.read_text())
    run = record["runs"][0]
    assert run["task_id"] == "anomaly_confirmation"
    messages = run["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert any(message["role"] == "assistant" for message in messages)
    # System message should include the dynamic tool catalog.
    assert "capture_thermal" in messages[0]["content"]
    assert "anomaly_confirmation" in messages[0]["content"]
