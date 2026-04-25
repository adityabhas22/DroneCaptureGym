"""Render rollouts as chat tokens and locate assistant turn boundaries.

The PPO trainer needs three things from each rollout:
    - input_ids:   the full conversation as model-ready tokens
    - response_mask: 1 on tokens we want to backprop through
                    (the model's emitted content for each assistant turn)
    - assistant_spans: per-turn (start, end) token positions, used to
                    place per-step rewards at the LAST token of each turn

This module produces all three from a sequence of chat messages. The
chat template applied here MUST match the one the rollout policy uses
during generation — both go through `tokenizer.apply_chat_template`, so
they do.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TokenizedTrajectory:
    """One rollout, ready for the PPO forward pass."""

    input_ids: torch.Tensor          # [T] int64
    attention_mask: torch.Tensor     # [T] int64
    response_mask: torch.Tensor      # [T] float32
    assistant_spans: list[tuple[int, int]]
    n_assistant_turns: int           # may be 0 if every turn was truncated

    @property
    def seq_len(self) -> int:
        return int(self.input_ids.shape[0])


def tokenize_trajectory(
    messages: list[dict[str, Any]],
    *,
    tokenizer,
    max_length: int,
    add_generation_prompt: bool = False,
    device: torch.device | str = "cpu",
) -> TokenizedTrajectory:
    """Render a chat as tokens with per-turn assistant span metadata.

    Walks the messages left-to-right; for each assistant turn, captures
    the (start, end) token position by tokenizing the prefix-with-
    generation-prompt (start) and the prefix-through-this-turn (end).
    Falls back gracefully when a tokenizer's `apply_chat_template`
    rejects `add_generation_prompt=True` for non-final messages.

    `max_length` truncates the trajectory by dropping turns that don't
    fit. We do NOT truncate mid-turn — that would corrupt the assistant
    span metadata. The cost is that overly long episodes contribute
    fewer turns to the gradient.
    """
    spans: list[tuple[int, int]] = []
    truncated = False

    # Sequentially compute prefix-token-counts to find each assistant boundary.
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        try:
            prefix_text = tokenizer.apply_chat_template(
                messages[:i],
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError):
            # Some templates require non-empty input — fall back to empty
            # prefix for the rare case the first message is the assistant.
            prefix_text = ""
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
        through_text = tokenizer.apply_chat_template(
            messages[: i + 1],
            tokenize=False,
            add_generation_prompt=False,
        )
        through_ids = tokenizer(through_text, add_special_tokens=False).input_ids

        start, end = len(prefix_ids), len(through_ids)
        if end > max_length:
            truncated = True
            break  # drop this and all subsequent turns
        spans.append((start, end))

    # Render the (possibly truncated) full sequence.
    if truncated and spans:
        # Include all messages up through the last surviving assistant turn.
        kept_assistant_count = len(spans)
        kept_messages: list[dict[str, Any]] = []
        seen_assistants = 0
        for msg in messages:
            kept_messages.append(msg)
            if msg.get("role") == "assistant":
                seen_assistants += 1
                if seen_assistants >= kept_assistant_count:
                    break
        text = tokenizer.apply_chat_template(
            kept_messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc.input_ids[0]
    attention_mask = enc.attention_mask[0]

    seq_len = int(input_ids.shape[0])
    # Drop any spans whose end exceeds the truncated seq_len; keep ones that fit.
    spans = [(s, e) for (s, e) in spans if e <= seq_len]

    response_mask = torch.zeros(seq_len, dtype=torch.float32)
    for s, e in spans:
        response_mask[s:e] = 1.0

    return TokenizedTrajectory(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        response_mask=response_mask.to(device),
        assistant_spans=spans,
        n_assistant_turns=len(spans),
    )


def pad_trajectories(
    trajectories: list[TokenizedTrajectory],
    *,
    pad_token_id: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Right-pad a batch of trajectories to a common length.

    Returns a dict with `input_ids`, `attention_mask`, and `response_mask`
    all shaped [B, T_max]. Padding tokens have mask=0.
    """
    if not trajectories:
        raise ValueError("cannot pad an empty trajectory batch")
    max_len = max(t.seq_len for t in trajectories)
    batch = len(trajectories)
    input_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    response_mask = torch.zeros((batch, max_len), dtype=torch.float32, device=device)
    for b, traj in enumerate(trajectories):
        L = traj.seq_len
        input_ids[b, :L] = traj.input_ids
        attention_mask[b, :L] = traj.attention_mask
        response_mask[b, :L] = traj.response_mask
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "response_mask": response_mask,
    }


__all__ = [
    "TokenizedTrajectory",
    "pad_trajectories",
    "tokenize_trajectory",
]
