"""Shared agent harness for inference, SFT data generation, and RL training.

The same primitives serve all three pipelines so they cannot disagree at
deployment time:

- `SYSTEM_PROMPT` and `render_system_prompt` — the role + tool-format spec.
- `openai_tool_schemas` / `anthropic_tool_schemas` — function-calling schemas
  derived from the live ToolRegistry.
- `render_observation` — compact text view of one observation (~600 tokens).
- `parse_action` — accepts JSON-text completions and tool_calls lists.
- `Policy` / `AgentContext` — the runner-policy boundary.
- `RolloutRunner` — runs a policy through one episode and emits a
  `RolloutResult` that downstream consumers (SFT exporter, RL trainer,
  trace tooling) all read.
- `trajectory_to_messages` / `trajectory_to_chat_messages` — convert a
  rollout into chat-format messages for SFT or replay.

Importing from `dronecaptureops.agent` is the only public path; submodules
may be reorganised without breaking callers.
"""

from dronecaptureops.agent.messages import (
    build_assistant_message,
    build_system_message,
    build_tool_result_message,
    build_user_message,
    trajectory_to_messages,
)
from dronecaptureops.agent.observation import render_initial_observation, render_observation
from dronecaptureops.agent.parser import parse_action
from dronecaptureops.agent.oracle import TaskOraclePolicy
from dronecaptureops.agent.policies import (
    AgentContext,
    Policy,
    RandomPolicy,
    ScriptedPolicy,
    act,
)
from dronecaptureops.agent.spec_aware_policy import SpecAwareScriptedPolicy
from dronecaptureops.agent.prompts import (
    INTERFACE_VERSION,
    SYSTEM_PROMPT,
    render_system_prompt,
)
from dronecaptureops.agent.rollout import (
    RolloutResult,
    RolloutRunner,
    RolloutStep,
    trajectory_to_chat_messages,
)
from dronecaptureops.agent.comparison import (
    ComparisonRequest,
    ComparisonResult,
    ModelRunSpec,
    ModelRunSummary,
    build_policy_for_spec,
    run_model_comparison,
)
from dronecaptureops.agent.schemas import (
    anthropic_tool_schemas,
    openai_tool_schemas,
)


def __getattr__(name):  # noqa: ANN001 - PEP 562 lazy import shim
    """Lazy-load LLM policies so the harness works without openai/anthropic/transformers/vllm."""

    if name in {"OpenAIChatPolicy", "AnthropicMessagesPolicy", "LocalHFPolicy"}:
        from dronecaptureops.agent import llm_policies

        return getattr(llm_policies, name)
    if name in {"VLLMEngine", "VLLMPolicy"}:
        from dronecaptureops.agent import vllm_policy

        return getattr(vllm_policy, name)
    if name in {"HFInferencePolicy", "HFInferenceTurnRecord", "HF_DEFAULT_BASE_URL"}:
        from dronecaptureops.agent import hf_inference_policy

        return getattr(hf_inference_policy, name)
    raise AttributeError(f"module 'dronecaptureops.agent' has no attribute {name!r}")


__all__ = [
    "INTERFACE_VERSION",
    "SYSTEM_PROMPT",
    "AgentContext",
    "ComparisonRequest",
    "ComparisonResult",
    "Policy",
    "ModelRunSpec",
    "ModelRunSummary",
    "RandomPolicy",
    "RolloutResult",
    "RolloutRunner",
    "RolloutStep",
    "ScriptedPolicy",
    "SpecAwareScriptedPolicy",
    "TaskOraclePolicy",
    "act",
    "anthropic_tool_schemas",
    "build_policy_for_spec",
    "build_assistant_message",
    "build_system_message",
    "build_tool_result_message",
    "build_user_message",
    "openai_tool_schemas",
    "parse_action",
    "render_initial_observation",
    "render_observation",
    "render_system_prompt",
    "run_model_comparison",
    "trajectory_to_chat_messages",
    "trajectory_to_messages",
]
