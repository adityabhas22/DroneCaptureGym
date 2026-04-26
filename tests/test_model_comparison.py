from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from dronecaptureops.agent import ComparisonRequest, ModelRunSpec, run_model_comparison
from dronecaptureops.agent.llm_policies import OpenAIChatPolicy
from dronecaptureops.agent.policies import AgentContext
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


def test_model_comparison_runs_scripted_vs_random_without_external_deps():
    request = ComparisonRequest(
        specs=[
            ModelRunSpec(name="scripted-baseline", policy="scripted"),
            ModelRunSpec(name="random-baseline", policy="random"),
        ],
        task_id="basic_thermal_survey",
        seed=7,
        max_steps=30,
        user_instruction="Inspect the assigned solar rows and submit grounded evidence.",
    )

    result = run_model_comparison(request)

    assert [summary.name for summary in result.summaries] == ["scripted-baseline", "random-baseline"]
    scripted = result.summaries[0]
    random = result.summaries[1]

    assert scripted.success is True
    assert scripted.final_reward >= 0.95
    assert scripted.action_sequence[:3] == ["get_mission_checklist", "list_assets", "takeoff"]
    assert scripted.final_report["submitted"] is True
    assert scripted.captures
    assert scripted.parse_errors == []

    assert random.seed == 7
    assert random.task_id == "basic_thermal_survey"
    assert len(random.action_sequence) == random.steps
    assert isinstance(random.reward_breakdown, dict)


def test_scripted_only_comparison_can_include_trace_and_rollout_artifacts():
    request = ComparisonRequest(
        specs=[ModelRunSpec(name="scripted", policy="scripted")],
        task_id="basic_thermal_survey",
        seed=7,
        max_steps=30,
        include_trace_artifacts=True,
        include_rollouts=True,
    )

    result = run_model_comparison(request)
    summary = result.summaries[0]

    assert summary.rollout is not None
    assert summary.rollout["task_id"] == "basic_thermal_survey"
    assert summary.trace_artifacts is not None
    assert "trace" in summary.trace_artifacts
    assert "trace_markdown" in summary.trace_artifacts
    assert summary.trace_artifacts["inspection_report"]["submitted"] is True


def test_model_run_spec_accepts_lazy_external_policy_shapes():
    request = ComparisonRequest(
        specs=[
            ModelRunSpec(name="openai-base", policy="openai", model="gpt-4o-mini"),
            ModelRunSpec(name="anthropic-base", policy="anthropic", model="claude-haiku-4-5-20251001"),
            ModelRunSpec(name="hf-router-base", policy="hf", model="Qwen/Qwen3-14B-Instruct-2507"),
            ModelRunSpec(name="local-base", policy="local_hf", base_model="Qwen/Qwen2.5-7B-Instruct"),
            ModelRunSpec(
                name="vllm-sft",
                policy="vllm",
                base_model="Qwen/Qwen2.5-7B-Instruct",
                adapter_path="artifacts/sft/final",
            ),
            ModelRunSpec(
                name="vllm-rl1",
                policy="vllm",
                base_model="Qwen/Qwen2.5-7B-Instruct",
                adapter_path="artifacts/ppo/rl1",
            ),
        ],
        task_id="basic_thermal_survey",
        seed=11,
        max_steps=5,
    )

    assert [spec.policy for spec in request.specs] == [
        "openai",
        "anthropic",
        "hf",
        "local_hf",
        "vllm",
        "vllm",
    ]
    assert request.specs[-1].adapter_path == "artifacts/ppo/rl1"


def test_openai_policy_accepts_gpt_token_alias(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_KEY", raising=False)
    monkeypatch.delenv("GPT_API_KEY", raising=False)
    monkeypatch.setenv("GPT_TOKEN", "gpt_fake_token_xxx")

    env = DroneCaptureOpsEnvironment()
    env.reset(seed=7)
    with patch("openai.OpenAI") as mock_openai:
        OpenAIChatPolicy(env=env, model="gpt-5.4-mini")

    _, kwargs = mock_openai.call_args
    assert kwargs["api_key"] == "gpt_fake_token_xxx"


def test_openai_policy_uses_max_completion_tokens_for_gpt5(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai_fake_token_xxx")

    env = DroneCaptureOpsEnvironment()
    observation = env.reset(seed=7)
    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        SimpleNamespace(
                            id="call_1",
                            function=SimpleNamespace(name="get_mission_checklist", arguments="{}"),
                        )
                    ],
                )
            )
        ]
    )

    with patch("openai.OpenAI", return_value=fake_client):
        policy = OpenAIChatPolicy(env=env, model="gpt-5.4-mini")
        action = policy.next_action(observation, AgentContext())

    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert action.tool_name == "get_mission_checklist"
    assert kwargs["max_completion_tokens"] == 1024
    assert "max_tokens" not in kwargs


def test_openai_policy_stores_prior_tool_call_as_plain_assistant_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai_fake_token_xxx")

    env = DroneCaptureOpsEnvironment()
    observation = env.reset(seed=7)
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(name="get_mission_checklist", arguments="{}"),
                            )
                        ],
                    )
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_2",
                                function=SimpleNamespace(name="list_assets", arguments="{}"),
                            )
                        ],
                    )
                )
            ]
        ),
    ]

    with patch("openai.OpenAI", return_value=fake_client):
        policy = OpenAIChatPolicy(env=env, model="gpt-5.4-mini")
        first = policy.next_action(observation, AgentContext())
        second = policy.next_action(observation, AgentContext())

    second_messages = fake_client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assistant_turns = [message for message in second_messages if message["role"] == "assistant"]
    assert first.tool_name == "get_mission_checklist"
    assert second.tool_name == "list_assets"
    assert assistant_turns
    assert "tool_calls" not in assistant_turns[-1]
