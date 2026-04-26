"""HF transformers ``model.generate()`` policy for GRPO rollouts.

Why a separate policy from ``VLLMPolicy``:

- The PPO branch hit repeated infra blockers in the colocated-vLLM path
  (V1 thread-safety, H200 Fabric Manager error 802, V1 ZMQ socket
  corruption under multi-thread rollouts, L40S OOM from 4B + vLLM
  colocation). All of those live in vLLM, not in PPO math.
- GRPO does not need a value head, which makes it cheap to skip vLLM
  entirely: rollouts can run directly through ``model.generate()`` on
  the same in-process model. Slower per-step (no continuous batching,
  no PagedAttention) but completely stable.

This policy mirrors ``VLLMPolicy``'s message construction (system prompt
+ tool schema + per-turn user observations) so the same chat template
applies during generation and during PPO/GRPO log-prob computation. We
only swap the generation backend.

Single-threaded by design. The GRPO rollout pool runs episodes
sequentially; there is no shared engine state to race on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from dronecaptureops.agent.messages import (
    build_system_message,
    build_user_message,
)
from dronecaptureops.agent.parser import parse_action
from dronecaptureops.agent.policies import AgentContext
from dronecaptureops.agent.schemas import openai_tool_schemas
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.tasks.solar_tasks import get_solar_task


@dataclass
class HFGeneratePolicy:
    """One-per-episode policy backed by an HF transformers model.

    The trainer owns a single ``model`` and ``tokenizer`` instance for
    the entire run; each ``HFGeneratePolicy`` borrows references and
    builds its own conversation history. ``model.generate()`` is called
    on a single prompt at a time (we do not need cross-rollout batching
    inside one episode — the GRPO step batches differently).

    The trainer activates / deactivates the LoRA adapter externally; we
    do not manage adapter state here. That is intentional so the same
    in-process model can serve both rollout (current LoRA) and reference
    (LoRA disabled via PEFT ``disable_adapter``) forwards.
    """

    model: Any
    tokenizer: Any
    env: DroneCaptureOpsEnvironment
    task_id: str | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 512
    max_history_steps: int = 12
    enable_thinking: bool = False
    name: str = "hf_generate"
    _messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _initialised: bool = field(default=False, init=False)

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Full chat history including raw assistant outputs.

        Read after a rollout completes — the trainer feeds this into
        ``tokenize_trajectory`` for the GRPO forward pass so log-probs
        are computed on EXACTLY the text the model emitted.
        """
        return list(self._messages)

    def __post_init__(self) -> None:
        self._tool_schemas = openai_tool_schemas(self.env._tools)  # noqa: SLF001

    # ---------- Policy protocol ----------

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        self._ensure_initialised()
        is_initial = len(self._messages) == 1
        self._messages.append(build_user_message(observation, is_initial=is_initial))

        prompt = self._render_prompt(self._trimmed())
        completion = self._generate(prompt)
        # Persist the raw assistant text so subsequent turns see what the
        # model actually emitted (including any prose preamble).
        self._messages.append({"role": "assistant", "content": completion})

        try:
            return parse_action(completion)
        except ActionValidationError:
            raise

    # ---------- Generation ----------

    def _generate(self, prompt: str) -> str:
        import torch

        ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = ids["input_ids"].to(self.model.device)
        attention_mask = ids.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=self.max_tokens,
            do_sample=self.temperature > 0.0,
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        prev_training = self.model.training
        self.model.eval()
        try:
            with torch.inference_mode():
                output = self.model.generate(input_ids=input_ids, **gen_kwargs)
        finally:
            self.model.train(prev_training)

        new_tokens = output[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        # Trim a known stop marker if the tokenizer left it in.
        for stop in ("<|im_end|>", "<|endoftext|>"):
            if stop in text:
                text = text.split(stop, 1)[0]
        return text

    def _render_prompt(self, messages: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is False:
            kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("chat_template_kwargs", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _ensure_initialised(self) -> None:
        if self._initialised:
            return
        registry = self.env._tools  # noqa: SLF001
        world = self.env.debug_world
        task = None
        if self.task_id:
            try:
                task = get_solar_task(self.task_id)
            except ValueError:
                task = None
        system_msg = build_system_message(registry=registry, world=world, task=task)
        schema_blob = json.dumps(self._tool_schemas, indent=2, sort_keys=True)
        system_msg = {
            "role": "system",
            "content": (
                system_msg["content"]
                + "\n\n# Tool JSON Schemas (for reference)\n```json\n"
                + schema_blob
                + "\n```\n"
            ),
        }
        self._messages = [system_msg]
        self._initialised = True

    def _trimmed(self) -> list[dict[str, Any]]:
        """Keep system prompt + first user turn + last K user/assistant pairs."""

        if len(self._messages) <= 1 + 2 * self.max_history_steps:
            return list(self._messages)
        head = self._messages[:2]
        tail = self._messages[-(2 * self.max_history_steps - 1):]
        return head + tail


__all__ = ["HFGeneratePolicy"]
