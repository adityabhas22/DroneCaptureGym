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
    _last_training_messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _pending_prompt_messages: list[dict[str, Any]] | None = field(default=None, init=False)
    _initialised: bool = field(default=False, init=False)

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Full chat history including raw assistant outputs.

        Read after a rollout completes — the trainer feeds this into
        ``tokenize_trajectory`` for the GRPO forward pass so log-probs
        are computed on EXACTLY the text the model emitted.
        """
        return list(self._messages)

    @property
    def training_messages(self) -> list[dict[str, Any]]:
        """Bounded prompt+completion window for policy-gradient training.

        Rollout generation uses ``_trimmed()`` to keep context bounded. The
        trainer must score the same bounded conversation, not the full episode
        history, otherwise long rollouts can lose every assistant span during
        tokenization and skip the GRPO update.
        """

        if self._last_training_messages:
            return list(self._last_training_messages)
        return self.messages

    def __post_init__(self) -> None:
        self._tool_schemas = openai_tool_schemas(self.env._tools)  # noqa: SLF001

    # ---------- Policy protocol ----------

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        """Single-call entrypoint: build prompt, run generate, parse action.

        Used by callers that don't batch across rollouts (the original
        Policy protocol). The group-batched rollout pool drives the same
        state transitions via the explicit ``prepare_prompt`` /
        ``ingest_completion`` pair below so it can fuse G ``generate()``
        calls into one.
        """

        prompt = self.prepare_prompt(observation, context)
        completion = self._generate(prompt)
        return self.ingest_completion(completion)

    # ---------- Group-batched protocol ----------

    def prepare_prompt(self, observation: DroneObservation, context: AgentContext) -> str:
        """Append the user turn for this observation and render its prompt.

        Mutates ``_messages`` (adds the user turn) and snapshots the
        bounded prompt window in ``_pending_prompt_messages`` so that the
        matching ``ingest_completion`` call records the same window into
        ``_last_training_messages``. Calling ``prepare_prompt`` twice in
        a row without an ``ingest_completion`` is a programming error and
        will raise.
        """

        self._ensure_initialised()
        if self._pending_prompt_messages is not None:
            raise RuntimeError(
                "HFGeneratePolicy.prepare_prompt called twice without "
                "ingest_completion in between — caller is mis-driving "
                "the batched protocol."
            )
        is_initial = len(self._messages) == 1
        self._messages.append(build_user_message(observation, is_initial=is_initial))
        prompt_messages = self._trimmed()
        self._pending_prompt_messages = list(prompt_messages)
        return self._render_prompt(prompt_messages)

    def ingest_completion(self, completion: str) -> RawDroneAction:
        """Record the assistant turn for the matching ``prepare_prompt`` call.

        Updates the full history (``_messages``) and the bounded
        training window (``_last_training_messages``) so the trainer
        can score exactly the prompt+completion pair the model saw.
        Parses and returns the next action; ``ActionValidationError``
        bubbles up so the runner can record the parse failure.
        """

        if self._pending_prompt_messages is None:
            raise RuntimeError(
                "HFGeneratePolicy.ingest_completion called without a "
                "matching prepare_prompt — caller is mis-driving the "
                "batched protocol."
            )
        prompt_messages = self._pending_prompt_messages
        self._pending_prompt_messages = None
        assistant_msg = {"role": "assistant", "content": completion}
        self._messages.append(assistant_msg)
        self._last_training_messages = [*prompt_messages, assistant_msg]
        return parse_action(completion)

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

    def render_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Public helper — used by the group rollout pool to re-render."""

        return self._render_prompt(messages)

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
                + ("\n\n/no_think" if self.enable_thinking is False else "")
                + "\n\n# Tool JSON Schemas (for reference)\n```json\n"
                + schema_blob
                + "\n```\n"
            ),
        }
        self._messages = [system_msg]
        self._initialised = True

    def _trimmed(self) -> list[dict[str, Any]]:
        """Keep system prompt + the most recent bounded interaction window.

        ``max_history_steps=1`` must mean "current observation only" during
        GRPO scoring. Keeping the first observation as well can push the
        assistant span past ``max_total_length`` and produce zero trainable
        trajectories.
        """

        if len(self._messages) <= 1 + 2 * self.max_history_steps:
            return list(self._messages)
        history_slots = max(1, 2 * self.max_history_steps - 1)
        return [self._messages[0], *self._messages[-history_slots:]]


__all__ = ["HFGeneratePolicy"]
