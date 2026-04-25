"""LLM-backed policy adapters: OpenAI, Anthropic, and local HF chat models.

All three implement the same `Policy` protocol as `ScriptedPolicy` /
`RandomPolicy`, so they're swappable at the rollout boundary. Each adapter:

1. Builds the system prompt (with the live tool catalog and task header)
   exactly once per episode.
2. Re-renders the most recent observation as a compact user message each
   step; the model's chat history grows by one user/assistant pair per
   turn (capped by `max_history_steps` to keep context bounded).
3. Parses the model's reply via the shared `parse_action`, so model output
   in JSON-text or tool_calls form both produce the same RawDroneAction.

External dependencies (`openai`, `anthropic`, `transformers`) are imported
lazily so the harness itself works without any of them installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.tasks.solar_tasks import get_solar_task

from dronecaptureops.agent.messages import (
    build_assistant_message,
    build_system_message,
    build_user_message,
)
from dronecaptureops.agent.parser import parse_action
from dronecaptureops.agent.policies import AgentContext
from dronecaptureops.agent.schemas import anthropic_tool_schemas, openai_tool_schemas


if TYPE_CHECKING:  # pragma: no cover
    from dronecaptureops.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Common base
# ---------------------------------------------------------------------------


@dataclass
class _LLMPolicyBase:
    """Shared per-episode bookkeeping for LLM policies.

    The first time `next_action` is called we lock in the system message
    (which depends on the tool registry + active task at reset time).
    Subsequent calls only append user/assistant turns. This mirrors the
    pattern in ClaimsGym/OpsArena that worked.
    """

    env: DroneCaptureOpsEnvironment
    task_id: str | None = None
    max_history_steps: int = 12
    name: str = "llm"
    _messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _initialised: bool = field(default=False, init=False)

    def _ensure_initialised(self) -> None:
        if self._initialised:
            return
        registry = self.env._tools  # noqa: SLF001 — public-ish read
        world = self.env.debug_world
        task = None
        if self.task_id:
            try:
                task = get_solar_task(self.task_id)
            except ValueError:
                task = None
        self._messages = [build_system_message(registry=registry, world=world, task=task)]
        self._initialised = True

    def _append_user(self, observation: DroneObservation) -> None:
        is_initial = len(self._messages) == 1
        self._messages.append(build_user_message(observation, is_initial=is_initial))

    def _append_assistant(self, action: RawDroneAction, *, use_tool_calls: bool) -> None:
        self._messages.append(build_assistant_message(action, use_tool_calls=use_tool_calls))

    def _trimmed(self) -> list[dict[str, Any]]:
        """Keep system + first turn + last `max_history_steps` user/assistant pairs."""

        if len(self._messages) <= 1 + 2 * self.max_history_steps:
            return list(self._messages)
        head = self._messages[:2]
        tail = self._messages[-(2 * self.max_history_steps - 1):]
        return head + tail


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@dataclass
class OpenAIChatPolicy(_LLMPolicyBase):
    """Calls OpenAI chat completions with native function-calling tool_calls."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1024
    api_base_url: str | None = None
    api_key: str | None = None
    use_tool_calls: bool = True
    name: str = "openai"

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised only in inference envs
            raise SystemExit("OpenAIChatPolicy requires `pip install openai`.") from exc
        self._client = OpenAI(
            base_url=self.api_base_url or os.getenv("OPENAI_API_BASE_URL") or os.getenv("API_BASE_URL"),
            api_key=self.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or "missing",
        )
        self._tool_schemas = openai_tool_schemas(self.env._tools)  # noqa: SLF001

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        self._ensure_initialised()
        self._append_user(observation)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._trimmed(),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.use_tool_calls:
            kwargs["tools"] = self._tool_schemas
            kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        action = self._parse_response(message)
        self._messages.append(_message_to_dict(message))
        return action

    def _parse_response(self, message: Any) -> RawDroneAction:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            payload = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
            return parse_action(payload)
        content = message.content or ""
        if not content.strip():
            raise ActionValidationError("model returned no tool_call and empty content")
        return parse_action(content)


def _message_to_dict(message: Any) -> dict[str, Any]:
    """Serialize an OpenAI ChatCompletionMessage back into a dict."""

    payload: dict[str, Any] = {"role": getattr(message, "role", "assistant")}
    content = getattr(message, "content", None)
    payload["content"] = content if content is not None else ""
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return payload


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------


@dataclass
class AnthropicMessagesPolicy(_LLMPolicyBase):
    """Calls Anthropic messages API with native tool_use."""

    model: str = "claude-haiku-4-5-20251001"
    temperature: float = 0.2
    max_tokens: int = 1024
    api_key: str | None = None
    use_tool_calls: bool = True
    name: str = "anthropic"

    def __post_init__(self) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("AnthropicMessagesPolicy requires `pip install anthropic`.") from exc
        self._client = Anthropic(api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._tool_schemas = anthropic_tool_schemas(self.env._tools)  # noqa: SLF001

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        self._ensure_initialised()
        self._append_user(observation)
        # Anthropic API splits system from messages.
        system_msg = self._messages[0]["content"]
        history = self._trimmed()[1:]
        response = self._client.messages.create(
            model=self.model,
            system=system_msg,
            messages=_to_anthropic_messages(history),
            tools=self._tool_schemas if self.use_tool_calls else None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        action = self._parse_response(response)
        self._messages.append(_anthropic_response_to_message(response))
        return action

    def _parse_response(self, response: Any) -> RawDroneAction:
        tool_uses = [block for block in response.content if getattr(block, "type", None) == "tool_use"]
        if tool_uses:
            block = tool_uses[0]
            return parse_action([{"type": "tool_use", "name": block.name, "input": block.input}])
        text_blocks = [block for block in response.content if getattr(block, "type", None) == "text"]
        if not text_blocks:
            raise ActionValidationError("anthropic response had no tool_use or text blocks")
        return parse_action(text_blocks[0].text)


def _to_anthropic_messages(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert chat-format history into Anthropic's content-block schema."""

    out: list[dict[str, Any]] = []
    for message in history:
        role = message["role"]
        if role == "user":
            out.append({"role": "user", "content": message["content"]})
        elif role == "assistant":
            blocks: list[dict[str, Any]] = []
            content = message.get("content") or ""
            if content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                args_raw = fn.get("arguments")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": call.get("id", "call_x"), "name": fn.get("name", ""), "input": args})
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
    return out


def _anthropic_response_to_message(response: Any) -> dict[str, Any]:
    blocks = []
    text_parts = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            blocks.append(
                {
                    "id": block.id,
                    "type": "function",
                    "function": {"name": block.name, "arguments": json.dumps(block.input)},
                }
            )
    payload: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
    if blocks:
        payload["tool_calls"] = blocks
    return payload


# ---------------------------------------------------------------------------
# Local HuggingFace chat model
# ---------------------------------------------------------------------------


@dataclass
class LocalHFPolicy(_LLMPolicyBase):
    """Run an HF transformers chat model locally via apply_chat_template + generate.

    Designed for evaluating SFT / GRPO / PPO checkpoints. Handles both
    JSON-text and tool_calls dialects via the shared parser.
    """

    model: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_id: str | None = None
    temperature: float = 0.7
    max_new_tokens: int = 512
    device: str = "auto"
    trust_remote_code: bool = False
    name: str = "hf"

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("LocalHFPolicy requires `pip install transformers torch`.") from exc
        tokenizer_id = self.tokenizer_id or self.model
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=self.trust_remote_code)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        self._ensure_initialised()
        self._append_user(observation)
        chat = self._trimmed()
        prompt = self._tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature if self.temperature > 0 else 1.0,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        action = parse_action(text)
        self._messages.append({"role": "assistant", "content": text})
        return action
