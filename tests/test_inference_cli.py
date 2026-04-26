"""Smoke tests for the inference CLI and offline policy adapters."""

from __future__ import annotations

import json
import re
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


def _lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.strip()]


def test_inference_cli_scripted_basic_thermal_survey():
    rc, stdout, stderr = _run("--task", "basic_thermal_survey", "--policy", "scripted")
    assert rc == 0, stderr
    lines = _lines(stdout)
    assert lines[0] == "[START] task=basic_thermal_survey env=dronecaptureops-gym model=scripted"
    assert any(line.startswith("[STEP] step=1 action=") for line in lines)
    assert lines[-1].startswith("[END] success=true ")
    score_match = re.search(r"score=(?P<score>-?\d+\.\d{2})\b", lines[-1])
    assert score_match is not None
    assert float(score_match.group("score")) >= 0.95
    assert "metadata" not in "\n".join(lines)


def test_inference_cli_random_runs_without_crash():
    rc, stdout, _ = _run("--task", "anomaly_confirmation", "--policy", "random", "--max-steps", "4", "--seed", "11")
    assert rc == 0
    lines = _lines(stdout)
    assert lines[0] == "[START] task=anomaly_confirmation env=dronecaptureops-gym model=random"
    step_lines = [line for line in lines if line.startswith("[STEP]")]
    assert len(step_lines) <= 4
    assert lines[-1].startswith("[END] ")


def test_inference_cli_runs_default_submission_tasks():
    rc, stdout, stderr = _run("--policy", "scripted")
    assert rc == 0, stderr
    starts = [line for line in _lines(stdout) if line.startswith("[START]")]
    assert starts == [
        "[START] task=basic_thermal_survey env=dronecaptureops-gym model=scripted",
        "[START] task=anomaly_confirmation env=dronecaptureops-gym model=scripted",
        "[START] task=audit_grade_strict_grounding env=dronecaptureops-gym model=scripted",
    ]


def test_inference_cli_writes_trajectory_and_messages_to_disk(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    messages_path = tmp_path / "messages.json"
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

    trace_record = json.loads(trace_path.read_text().splitlines()[0])
    assert trace_record["task_id"] == "anomaly_confirmation"
    assert trace_record["policy"] == "scripted"
    assert trace_record["success"] is True

    payload = json.loads(messages_path.read_text())
    record = payload["runs"][0]
    assert record["task_id"] == "anomaly_confirmation"
    messages = record["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert any(message["role"] == "assistant" for message in messages)
    # System message should include the dynamic tool catalog.
    assert "capture_thermal" in messages[0]["content"]
    assert "anomaly_confirmation" in messages[0]["content"]
