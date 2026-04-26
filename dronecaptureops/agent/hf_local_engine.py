"""In-process HF transformers rollout engine — vLLM replacement.

Why this exists: vLLM 0.11.x–0.20.x V1 engine has an unfixed wire-protocol
race in `process_input_sockets` that drops requests under our LoRA-hot-swap
+ multi-rollout workload. The bug is documented in vllm-project/vllm PRs
#25198 and #26196 (both closed without fix). Manifests as the worker thread
crashing on a malformed type frame (`b'\\x00\\x00'` or `b'READY'`), losing
all in-flight rollouts → trainer deadlocks at `future.result()`.

Replacement uses the trainer's already-loaded PEFT model directly:
- No separate engine process → no IPC, no spawn, no fabric init issues
- No LoRA hot-swap → engine sees live training weights for free
- No double-VRAM cost (vLLM normally holds its own copy of the weights)

Throughput strategy: a coalescing batcher collects concurrent generate()
calls from rollout-pool threads and dispatches them as one batched
`model.generate()`. With Qwen3-4B BF16 + Flash Attention 2 on H200, batched
generate hits ~500–1000 tok/s aggregate for batches of 8–16 — enough to
finish a 50-step PPO sweep inside our remaining budget.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from dronecaptureops.agent.messages import (
    build_assistant_message,  # noqa: F401  (re-exported for symmetry)
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

import json


LOG = logging.getLogger("dronecaptureops.agent.hf_local_engine")


@dataclass
class _PendingRequest:
    prompt: str
    temperature: float
    top_p: float
    max_tokens: int
    stop: tuple[str, ...]
    event: threading.Event = field(default_factory=threading.Event)
    result: list[str] = field(default_factory=lambda: [""])
    error: list[BaseException | None] = field(default_factory=lambda: [None])


class HFLocalEngine:
    """Coalescing-batch wrapper over a trainer-owned HF causal LM.

    Constructed by the trainer with a reference to its own PEFT-wrapped
    model and tokenizer. Rollout-pool worker threads call `generate()`
    concurrently; a background dispatcher thread coalesces up to
    `max_batch_size` pending requests every `batch_interval_s` seconds
    (or as soon as the batch is full) and runs ONE `model.generate()`
    call for the whole batch. This is the same continuous-batching idea
    vLLM implements, just without the buggy IPC layer.

    Thread safety:
    - The dispatcher always runs `model.generate()` under `model.eval()`
      and `torch.no_grad()`.
    - A `gen_lock` serializes generate() vs. any concurrent training
      forward/backward (the trainer should hold this lock during PPO
      updates if it wants to protect the model — but in our flow,
      rollouts and training alternate sequentially per PPO step, so
      contention is rare).
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: Any,
        max_batch_size: int = 16,
        batch_interval_s: float = 0.05,
        max_history_steps: int = 12,
        enable_thinking: bool = False,
    ) -> None:
        import torch

        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.batch_interval_s = batch_interval_s
        self.enable_thinking = enable_thinking
        self.max_history_steps = max_history_steps

        # tokenizer must have a pad token for batched generation
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        # Left-padding is required for batched generation so all sequences
        # end at the same position (the generation start point).
        tokenizer.padding_side = "left"

        self._pending: list[_PendingRequest] = []
        self._pending_lock = threading.Lock()
        self._cv = threading.Condition(self._pending_lock)
        self._stop = False
        self.gen_lock = threading.Lock()

        self._dispatcher = threading.Thread(target=self._dispatch_loop, daemon=True, name="HFLocalEngine-dispatch")
        self._dispatcher.start()

    # ----- Prompt rendering (drop-in for VLLMEngine) -----

    def render_prompt(self, messages: list[dict[str, Any]]) -> str:
        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if self.enable_thinking is False:
            kwargs["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            return self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("chat_template_kwargs", None)
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    # ----- Public generate() (drop-in for VLLMEngine) -----

    def generate(
        self,
        prompts: list[str],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        stop: list[str] | None = None,
        lora_request: Any | None = None,  # accepted for interface compat; ignored
    ) -> list[str]:
        """Submit prompts to the batcher; block until all complete.

        `lora_request` is silently ignored — the engine uses the live
        trainable model directly, so there is no adapter to swap.
        """

        if not prompts:
            return []
        stop_tuple = tuple(stop or [])
        reqs = [
            _PendingRequest(prompt=p, temperature=temperature, top_p=top_p, max_tokens=max_tokens, stop=stop_tuple)
            for p in prompts
        ]
        with self._cv:
            self._pending.extend(reqs)
            self._cv.notify()
        # Wait for all to complete
        for r in reqs:
            r.event.wait()
            if r.error[0] is not None:
                raise r.error[0]
        return [r.result[0] for r in reqs]

    # ----- Internals -----

    def _dispatch_loop(self) -> None:
        while not self._stop:
            with self._cv:
                # Wait until at least one request is pending
                while not self._pending and not self._stop:
                    self._cv.wait(timeout=1.0)
                if self._stop:
                    return
                # Brief delay to allow more requests to accumulate
                if len(self._pending) < self.max_batch_size:
                    self._cv.wait(timeout=self.batch_interval_s)
                # Pull a batch (up to max_batch_size, all sharing same sampling params)
                batch = self._pop_compatible_batch()
            if not batch:
                continue
            try:
                self._run_batch(batch)
            except BaseException as exc:  # noqa: BLE001
                LOG.exception("HFLocalEngine batch failed")
                for r in batch:
                    r.error[0] = exc
                    r.event.set()

    def _pop_compatible_batch(self) -> list[_PendingRequest]:
        """Pop up to max_batch_size requests sharing identical sampling params.

        Different temperature/top_p/max_tokens can't be batched in one
        `model.generate()` call (HF generate uses one set of params per call),
        so we group by (temp, top_p, max_tokens, stop) tuple and dispatch
        one group per cycle.
        """
        if not self._pending:
            return []
        head = self._pending[0]
        key = (head.temperature, head.top_p, head.max_tokens, head.stop)
        batch: list[_PendingRequest] = []
        keep: list[_PendingRequest] = []
        for r in self._pending:
            if len(batch) < self.max_batch_size and (r.temperature, r.top_p, r.max_tokens, r.stop) == key:
                batch.append(r)
            else:
                keep.append(r)
        self._pending = keep
        return batch

    def _run_batch(self, batch: list[_PendingRequest]) -> None:
        torch = self._torch
        head = batch[0]
        prompts = [r.prompt for r in batch]

        # Tokenize with left-padding so all completions start at the same offset.
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(self.device)
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": head.max_tokens,
            "do_sample": head.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if head.temperature > 0:
            gen_kwargs["temperature"] = head.temperature
            gen_kwargs["top_p"] = head.top_p
        if head.stop:
            # HF supports stop_strings via the tokenizer kwarg
            gen_kwargs["stop_strings"] = list(head.stop)
            gen_kwargs["tokenizer"] = self.tokenizer

        n_in_flight = 0
        with self._pending_lock:
            n_in_flight = len(self._pending)
        LOG.info(
            "engine: dispatching batch=%d (prompt_len=%d, max_new=%d, in_flight_after=%d)",
            len(batch), prompt_len, head.max_tokens, n_in_flight,
        )
        t0 = time.perf_counter()

        with self.gen_lock:
            was_training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    out = self.model.generate(**enc, **gen_kwargs)
            finally:
                self.model.train(was_training)
        t_gen = time.perf_counter() - t0
        n_new = out.shape[1] - prompt_len
        total_new_tokens = n_new * len(batch)
        LOG.info(
            "engine: batch=%d done in %.1fs (%.0f tok/s aggregate, %d new tok/req)",
            len(batch), t_gen, total_new_tokens / max(t_gen, 1e-6), n_new,
        )

        # Decode the generated completion (everything after the prompt).
        # Note: with left-padding, prompt occupies the LAST `prompt_len` slots
        # of input_ids (after the leading pad tokens). model.generate returns
        # the full sequence (input + new). New tokens start at index `prompt_len`.
        new_tokens = out[:, prompt_len:]
        for r, ids in zip(batch, new_tokens):
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            # Strip any trailing copies of the stop strings
            for s in r.stop:
                if s and text.endswith(s):
                    text = text[: -len(s)]
            r.result[0] = text
            r.event.set()

    def shutdown(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify_all()


# ---------------------------------------------------------------------------
# Per-rollout policy — drop-in API replacement for VLLMPolicy
# ---------------------------------------------------------------------------


@dataclass
class HFLocalPolicy:
    """Per-episode policy backed by the shared HFLocalEngine."""

    engine: HFLocalEngine
    env: DroneCaptureOpsEnvironment
    task_id: str | None = None
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 1024
    max_history_steps: int = 12
    name: str = "hf_local"
    lora_request: Any | None = None  # accepted but ignored (no hot-swap)
    _messages: list[dict[str, Any]] = field(default_factory=list, init=False)
    _initialised: bool = field(default=False, init=False)

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self._messages)

    def __post_init__(self) -> None:
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
        )[0]

        self._messages.append({"role": "assistant", "content": completion})

        try:
            return parse_action(completion)
        except ActionValidationError:
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
        if len(self._messages) <= 1 + 2 * self.max_history_steps:
            return list(self._messages)
        head = self._messages[:2]
        tail = self._messages[-(2 * self.max_history_steps - 1):]
        return head + tail


__all__ = ["HFLocalEngine", "HFLocalPolicy"]
