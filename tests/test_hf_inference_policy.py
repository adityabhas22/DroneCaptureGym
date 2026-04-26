"""Tests for HFInferencePolicy with mocked OpenAI client.

We simulate every meaningful failure path the live HF endpoint can throw
(429 rate limit, 503 model loading, transient network error, malformed
output) so the retry logic and parse-error handling are covered without
hitting the network.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from dronecaptureops.agent import RolloutRunner
from dronecaptureops.agent.policies import AgentContext
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Helpers — build a fake OpenAI ChatCompletion-style response
# ---------------------------------------------------------------------------


def _fake_response(text: str, tool_calls: list | None = None, finish_reason: str = "stop") -> SimpleNamespace:
    """Build a SimpleNamespace that quacks like an OpenAI ChatCompletion."""

    message = SimpleNamespace(
        role="assistant",
        content=text,
        tool_calls=tool_calls or None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason, index=0)
    usage = SimpleNamespace(prompt_tokens=120, completion_tokens=40, total_tokens=160)
    return SimpleNamespace(choices=[choice], usage=usage, model="fake-model", id="resp_x")


def _fake_tool_call(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id="call_xyz",
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


class _FakeOpenAIError(Exception):
    """Mimics openai.RateLimitError / APIStatusError shape."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------------------
# Construction & auth
# ---------------------------------------------------------------------------


def test_hf_policy_requires_token_or_explicit_key(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    from dronecaptureops.agent import HFInferencePolicy  # type: ignore[attr-defined]

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    with pytest.raises(SystemExit, match="HF_TOKEN"):
        HFInferencePolicy(env=env, model="Qwen/Qwen3-14B-Instruct-2507")


def test_hf_policy_uses_hf_token_env_var(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token_xxx")
    from dronecaptureops.agent import HFInferencePolicy  # type: ignore[attr-defined]

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    with patch("openai.OpenAI") as mock_openai:
        HFInferencePolicy(env=env, model="Qwen/Qwen3-14B-Instruct-2507")
    args, kwargs = mock_openai.call_args
    assert kwargs["api_key"] == "hf_fake_token_xxx"
    assert kwargs["base_url"].endswith("/v1")


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def _build_policy_with_mock_client(env: DroneCaptureOpsEnvironment, mock_create) -> object:
    """Construct HFInferencePolicy with the OpenAI client's `create` mocked."""

    from dronecaptureops.agent import HFInferencePolicy  # type: ignore[attr-defined]

    fake_client = MagicMock()
    fake_client.chat.completions.create = mock_create
    with patch("openai.OpenAI", return_value=fake_client):
        policy = HFInferencePolicy(
            env=env,
            task_id="basic_thermal_survey",
            model="Qwen/Qwen3-14B-Instruct-2507",
            api_key="hf_fake",
            max_retries=3,
            initial_backoff_s=0.0,
            max_backoff_s=0.0,
        )
    return policy


def test_retries_on_429_rate_limit_then_succeeds():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="basic_thermal_survey")
    obs = env.step({"tool_name": "get_telemetry", "arguments": {}})

    create_mock = MagicMock(side_effect=[
        _FakeOpenAIError("Too many requests", 429),
        _FakeOpenAIError("rate limit hit", 429),
        _fake_response("", tool_calls=[_fake_tool_call("takeoff", {"altitude_m": 18})]),
    ])
    policy = _build_policy_with_mock_client(env, create_mock)
    action = policy.next_action(obs, AgentContext())
    assert action.tool_name == "takeoff"
    assert create_mock.call_count == 3
    # Retry count surfaced in the audit record.
    assert policy.turns[0].retries == 2


def test_retries_on_503_model_loading():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="basic_thermal_survey")
    obs = env.step({"tool_name": "get_telemetry", "arguments": {}})

    loading_error = _FakeOpenAIError("Model is currently loading; estimated_time=20s", 503)
    create_mock = MagicMock(side_effect=[
        loading_error,
        _fake_response("", tool_calls=[_fake_tool_call("takeoff", {"altitude_m": 18})]),
    ])
    policy = _build_policy_with_mock_client(env, create_mock)
    action = policy.next_action(obs, AgentContext())
    assert action.tool_name == "takeoff"
    assert create_mock.call_count == 2


def test_does_not_retry_on_401_auth_error():
    """A bad token should fail immediately, not exhaust retries."""

    from dronecaptureops.core.errors import ActionValidationError

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="basic_thermal_survey")
    obs = env.step({"tool_name": "get_telemetry", "arguments": {}})

    create_mock = MagicMock(side_effect=_FakeOpenAIError("Invalid token", 401))
    policy = _build_policy_with_mock_client(env, create_mock)
    with pytest.raises(ActionValidationError, match="Invalid token"):
        policy.next_action(obs, AgentContext())
    assert create_mock.call_count == 1
    assert policy.turns[0].finish_reason == "api_error"
    assert policy.turns[0].api_error["status_code"] == 401


def test_exhausts_retries_with_descriptive_error():
    from dronecaptureops.core.errors import ActionValidationError

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="basic_thermal_survey")
    obs = env.step({"tool_name": "get_telemetry", "arguments": {}})

    create_mock = MagicMock(side_effect=_FakeOpenAIError("Too many requests", 429))
    policy = _build_policy_with_mock_client(env, create_mock)
    with pytest.raises(ActionValidationError, match="exhausted .* retries"):
        policy.next_action(obs, AgentContext())
    # max_retries=3 ⇒ 3 attempts.
    assert create_mock.call_count == 3
    assert policy.turns[0].finish_reason == "api_error"


# ---------------------------------------------------------------------------
# Output parsing + audit
# ---------------------------------------------------------------------------


def test_parses_tool_calls_response():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="basic_thermal_survey")
    obs = env.step({"tool_name": "get_telemetry", "arguments": {}})

    create_mock = MagicMock(return_value=_fake_response(
        "", tool_calls=[_fake_tool_call("set_camera_source", {"source": "thermal"})]
    ))
    policy = _build_policy_with_mock_client(env, create_mock)
    action = policy.next_action(obs, AgentContext())
    assert action.tool_name == "set_camera_source"
    assert action.arguments == {"source": "thermal"}


def test_parses_json_text_content_when_no_tool_calls():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="basic_thermal_survey")
    obs = env.step({"tool_name": "get_telemetry", "arguments": {}})

    create_mock = MagicMock(return_value=_fake_response(
        '{"tool": "takeoff", "args": {"altitude_m": 22}}'
    ))
    policy = _build_policy_with_mock_client(env, create_mock)
    action = policy.next_action(obs, AgentContext())
    assert action.tool_name == "takeoff"
    assert action.arguments == {"altitude_m": 22}


def test_malformed_response_surfaces_as_action_validation_error():
    """No tool_calls + non-JSON content ⇒ ActionValidationError so the
    rollout runner records a parse error (not a silent success)."""

    from dronecaptureops.core.errors import ActionValidationError

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="basic_thermal_survey")
    obs = env.step({"tool_name": "get_telemetry", "arguments": {}})

    create_mock = MagicMock(return_value=_fake_response(
        "I'd really like to help but I'm not sure what to do here."
    ))
    policy = _build_policy_with_mock_client(env, create_mock)
    with pytest.raises(ActionValidationError):
        policy.next_action(obs, AgentContext())

    # The turn was still recorded for audit, with parse_error populated.
    assert len(policy.turns) == 1
    assert policy.turns[0].parse_error is not None
    assert "I'd really like to help" in policy.turns[0].response_text


def test_records_token_usage_per_turn():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7, task="basic_thermal_survey")
    obs = env.step({"tool_name": "get_telemetry", "arguments": {}})

    create_mock = MagicMock(return_value=_fake_response(
        "", tool_calls=[_fake_tool_call("takeoff", {"altitude_m": 18})]
    ))
    policy = _build_policy_with_mock_client(env, create_mock)
    policy.next_action(obs, AgentContext())
    turn = policy.turns[0]
    assert turn.prompt_tokens == 120
    assert turn.completion_tokens == 40
    assert turn.total_tokens == 160


# ---------------------------------------------------------------------------
# End-to-end: drive a full rollout via mocked HF
# ---------------------------------------------------------------------------


def test_rollout_runner_uses_mocked_hf_policy_for_one_episode():
    """Drive a full short rollout with mocked responses and verify the
    rollout runner records a parse error when the model output is bad."""

    env = DroneCaptureOpsEnvironment()
    runner = RolloutRunner(env=env)

    # Mock returns malformed text so we get parse_error every step.
    create_mock = MagicMock(return_value=_fake_response("hello!"))
    policy = _build_policy_with_mock_client(env, create_mock)
    policy.task_id = "basic_thermal_survey"

    result = runner.run(policy, seed=7, task_id="basic_thermal_survey", max_steps=3)
    assert result.steps == 3
    parse_errors = [step for step in result.trajectory if step.parse_error]
    assert len(parse_errors) == 3
    assert len(policy.turns) == 3


# ---------------------------------------------------------------------------
# eval_models.py CLI: offline provider end-to-end (no HF needed)
# ---------------------------------------------------------------------------


def test_eval_models_offline_provider_writes_full_diagnostic_jsonl(tmp_path):
    rows_path = tmp_path / "rows.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    summary_path = tmp_path / "summary.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "training.eval_models",
            "--provider", "offline",
            "--models", "task_oracle,random",
            "--tasks", "basic_thermal_survey,anomaly_confirmation",
            "--seeds", "1",
            "--output", str(rows_path),
            "--audit-trail", str(audit_path),
            "--summary-output", str(summary_path),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert rows_path.exists()
    assert audit_path.exists()
    assert summary_path.exists()

    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    # 2 models × 2 tasks × 1 seed = 4 cells
    assert len(rows) == 4
    for row in rows:
        # Diagnostic profile must be present per row.
        assert "failure_mode" in row
        assert "checkpoints" in row
        assert "tool_calls" in row
        assert "coverage" in row
        assert "reward_components" in row
        assert "oracle_comparison" in row
        # Oracle must always pass; random must always fail.
        if row["model"] == "task_oracle":
            assert row["success"] is True
            assert row["failure_mode"] == "success"
        else:
            assert row["success"] is False
            assert row["failure_mode"] != "success"

    # Audit trail must be populated for offline provider too (uses
    # trajectory steps as the fallback record format).
    audit_lines = audit_path.read_text().splitlines()
    assert audit_lines, "audit trail must not be empty"

    summary = json.loads(summary_path.read_text())
    assert "diagnostics" in summary
    assert "task_oracle" in summary["diagnostics"]
    assert "random" in summary["diagnostics"]
    assert summary["diagnostics"]["task_oracle"]["failure_mode_distribution"].get("success") == 1.0
