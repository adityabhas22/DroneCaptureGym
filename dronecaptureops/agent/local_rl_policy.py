"""Local-HF policy used by the online RL trainer.

Differs from `LocalHFPolicy` in two important ways:

1. It records, on every turn, the prompt token IDs the model conditioned on
   plus the response token IDs the model emitted. The trainer needs both to
   recompute log-probabilities under the current and reference policies for
   GRPO-style updates.
2. It is generation-only. Optimisation lives in `training/train_grpo.py`;
   this class never holds a gradient or an optimizer.

Heavyweight imports (torch, transformers) only happen inside `__post_init__`
so the rest of the agent harness keeps working without ML deps installed.
This mirrors the pattern used by `HFInferencePolicy` and `LocalHFPolicy`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dronecaptureops.agent.llm_policies import _LLMPolicyBase
from dronecaptureops.agent.parser import parse_action_with_thinking
from dronecaptureops.agent.policies import AgentContext
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import DroneObservation, RawDroneAction


if TYPE_CHECKING:  # pragma: no cover
    pass


LOG = logging.getLogger("dronecaptureops.agent.local_rl")


@dataclass
class RLTurnRecord:
    """Per-turn audit record consumed by the RL trainer.

    `prompt_token_ids` are the ids the model was conditioned on (after chat
    template rendering and trimming). `response_token_ids` are the ids the
    model emitted; together they form the (state, action) pair the trainer
    scores. The advantage is filled in later, once the episode terminates and
    the reward is known.
    """

    step: int
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    response_text: str
    parse_error: str | None
    thinking: str = ""
    finish_reason: str | None = None
    advantage: float = 0.0
    reward: float = 0.0


@dataclass
class LocalHFRLPolicy(_LLMPolicyBase):
    """Tokenizer-aware local HF policy that records (prompt, response) ids per turn.

    `model` and `tokenizer` are passed in pre-loaded so the trainer can share
    a single instance across many rollouts (loading a 4B model once is the
    main cost). Use `clear_turns()` between episodes; turns are kept in
    insertion order so the trainer can reconstruct each step's gradient.
    """

    model: Any = None
    tokenizer: Any = None
    max_new_tokens: int = 512
    temperature: float = 0.9
    top_p: float = 0.95
    do_sample: bool = True
    name: str = "local_rl"
    turns: list[RLTurnRecord] = field(default_factory=list, init=False)
    _device: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise ValueError("LocalHFRLPolicy requires preloaded `model` and `tokenizer`.")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("LocalHFRLPolicy requires `pip install torch transformers`.") from exc
        self._device = next(self.model.parameters()).device
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._torch = torch

    def clear_turns(self) -> None:
        """Reset per-episode bookkeeping. Call between rollouts."""

        self.turns = []
        self._messages = []
        self._initialised = False

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        self._ensure_initialised()
        is_initial = len(self._messages) == 1
        from dronecaptureops.agent.messages import build_user_message

        self._messages.append(build_user_message(observation, is_initial=is_initial))
        chat = self._trimmed()
        prompt_ids = self.tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self._device)

        with self._torch.no_grad():
            output = self.model.generate(
                input_ids=prompt_ids,
                attention_mask=self._torch.ones_like(prompt_ids),
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else 1.0,
                top_p=self.top_p if self.do_sample else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
            )
        sequences = output.sequences
        prompt_len = prompt_ids.shape[1]
        response_ids = sequences[0, prompt_len:].tolist()
        if response_ids and response_ids[-1] == self.tokenizer.eos_token_id:
            finish_reason = "stop"
        else:
            finish_reason = "length"
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        parse_error: str | None = None
        thinking = ""
        action: RawDroneAction | None = None
        try:
            parsed = parse_action_with_thinking(response_text)
            action = parsed.action
            thinking = parsed.thinking
        except ActionValidationError as exc:
            parse_error = str(exc)

        prompt_token_ids = prompt_ids[0].tolist()
        self.turns.append(
            RLTurnRecord(
                step=len(self.turns) + 1,
                prompt_token_ids=prompt_token_ids,
                response_token_ids=response_ids,
                response_text=response_text,
                parse_error=parse_error,
                thinking=thinking,
                finish_reason=finish_reason,
            )
        )
        self._messages.append({"role": "assistant", "content": response_text})

        if action is None:
            raise ActionValidationError(parse_error or "model produced no parseable action")
        return action


__all__ = ["LocalHFRLPolicy", "RLTurnRecord"]
