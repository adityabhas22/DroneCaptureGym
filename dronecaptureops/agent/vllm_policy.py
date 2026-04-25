"""vLLM-backed policy for high-throughput evaluation of HF chat models.

Why a separate adapter from `LocalHFPolicy`:
- Loading a 30-32B model with raw transformers takes 1-2 min and runs at
  ~20 tok/s on a single H200. Running a 45-episode eval matrix (15 tasks ×
  3 seeds) would take 4-8 hours per model that way.
- vLLM with paged attention + continuous batching loads once per process
  and inferences at 80-200+ tok/s. Same matrix takes ~30-60 min per model.

Architecture:
- `VLLMEngine` owns the shared vllm.LLM instance. One instance per
  process; reused across all episodes/tasks for a given model.
- `VLLMPolicy` is per-episode (it owns the conversation history). It
  borrows a reference to the engine and asks it to generate on demand.

The whole vllm dependency is imported lazily so this file can sit in the
package without forcing every consumer to install vllm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from dronecaptureops.agent.messages import (
    build_assistant_message,
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


class VLLMEngine:
    """Shared vLLM engine + tokenizer wrapper.

    Construct once per (model_id, sampling_args). Reuse across many
    `VLLMPolicy` instances so we don't reload the weights for every
    episode in the eval matrix.
    """

    def __init__(
        self,
        model: str,
        *,
        max_model_len: int = 32768,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        enforce_eager: bool = False,
        trust_remote_code: bool = False,
        download_dir: str | None = None,
        tensor_parallel_size: int = 1,
        enable_thinking: bool = False,
        enable_lora: bool = False,
        max_lora_rank: int = 64,
        max_loras: int = 1,
    ) -> None:
        try:
            from vllm import LLM
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "VLLMEngine requires `pip install vllm`. On the H200 host this "
                "needs the matching CUDA wheel; see vllm install docs."
            ) from exc

        self.model_id = model
        self.enable_thinking = enable_thinking
        self.enable_lora = enable_lora

        # Qwen3 chat templates respect a `chat_template_kwargs={'enable_thinking': False}`
        # toggle. We apply chat templates manually below so we can pass that
        # through cleanly.
        llm_kwargs: dict[str, Any] = dict(
            model=model,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            trust_remote_code=trust_remote_code,
            download_dir=download_dir,
            tensor_parallel_size=tensor_parallel_size,
        )
        # LoRA support — required for PPO LoRA hot-swap each training step.
        # `max_loras=1` keeps memory tight; we only ever serve the current
        # policy's adapter at a time.
        if enable_lora:
            llm_kwargs.update(
                enable_lora=True,
                max_lora_rank=max_lora_rank,
                max_loras=max_loras,
            )
        self._llm = LLM(**llm_kwargs)
        self._tokenizer = self._llm.get_tokenizer()

    def render_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Apply the model's chat template to a messages list."""

        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is False:
            kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return self._tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            # Older tokenizers don't accept chat_template_kwargs; fall back.
            kwargs.pop("chat_template_kwargs", None)
            return self._tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: list[str] | None = None,
        lora_request: Any | None = None,
    ) -> list[str]:
        """Generate completions for a batch of prompts. Returns text only.

        `lora_request` (a `vllm.lora.request.LoRARequest`) selects which
        adapter to apply for these prompts — required during PPO when
        the policy weights live in a LoRA adapter. Pass `None` to
        generate from the base model. Threads sharing this engine can
        pass different `lora_request`s safely; vLLM batches across them.
        """

        from vllm import SamplingParams

        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop or [],
        )
        kwargs: dict[str, Any] = {}
        if lora_request is not None:
            if not self.enable_lora:
                raise RuntimeError(
                    "VLLMEngine was constructed with enable_lora=False; "
                    "rebuild with enable_lora=True to use LoRA adapters."
                )
            kwargs["lora_request"] = lora_request
        outputs = self._llm.generate(prompts, params, **kwargs)
        return [output.outputs[0].text for output in outputs]


@dataclass
class VLLMPolicy:
    """One per-episode vLLM-backed policy.

    Pass the shared `VLLMEngine` so we don't reload weights. The policy
    itself just builds messages, asks the engine to generate, and parses
    the response through the same `parse_action` shared by every other
    policy.

    During PPO, set `lora_request` to point at the adapter the trainer
    just saved — the engine will apply that adapter for this rollout
    only. Different rollouts can use different adapters concurrently
    (vLLM batches across them), but in our single-step training pattern
    they all use the current step's adapter.
    """

    engine: VLLMEngine
    env: DroneCaptureOpsEnvironment
    task_id: str | None = None
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 1024
    max_history_steps: int = 12
    name: str = "vllm"
    lora_request: Any | None = None
    _messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _initialised: bool = field(default=False, init=False)

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Full chat history including raw assistant outputs.

        Read after a rollout completes — the trainer feeds this into
        `tokenize_trajectory` for the PPO forward pass so log-probs are
        computed on EXACTLY the text the model emitted (including any
        prose before the JSON tool call).
        """
        return list(self._messages)

    def __post_init__(self) -> None:
        # Generate a tools schema string we can append to the system prompt for
        # models that benefit from seeing the raw schema (Qwen3 picks it up
        # via the system message).
        self._tool_schemas = openai_tool_schemas(self.env._tools)  # noqa: SLF001

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        self._ensure_initialised()
        is_initial = len(self._messages) == 1
        self._messages.append(build_user_message(observation, is_initial=is_initial))

        prompt = self.engine.render_prompt(self._trimmed())
        completion = self.engine.generate(
            [prompt],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stop=["<|im_end|>", "<|endoftext|>"],
            lora_request=self.lora_request,
        )[0]

        # Persist the raw assistant text so subsequent turns see what the
        # model actually emitted (including any prose preamble).
        self._messages.append({"role": "assistant", "content": completion})

        try:
            return parse_action(completion)
        except ActionValidationError:
            # Re-raise so the runner records a parse error for this step.
            raise

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
        # Append a JSON schema reminder — models reproduce the contract more
        # reliably when they see the actual schema, even though our parser
        # accepts looser JSON-text.
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
        """Keep the system prompt + first user turn + last K user/assistant pairs."""

        if len(self._messages) <= 1 + 2 * self.max_history_steps:
            return list(self._messages)
        head = self._messages[:2]
        tail = self._messages[-(2 * self.max_history_steps - 1):]
        return head + tail


__all__ = ["VLLMEngine", "VLLMPolicy"]
