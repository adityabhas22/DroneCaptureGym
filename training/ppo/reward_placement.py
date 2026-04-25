"""Place per-step rewards at turn-end token positions.

The DroneCaptureGym reward is a clamped scalar in [-1, 1] returned per
env step. For token-level GAE we want the reward at the LAST token of
each assistant turn — preceding tokens get advantages via value
bootstrapping, the way the Practitioner's Guide to Multi-turn Agentic
RL recommends (arxiv 2510.01132 §6).

Conventions in this module:
- `assistant_spans` is a list of (start, end) token-index pairs aligned
  to the rendered chat sequence; one pair per assistant turn.
- `step_rewards` is the per-turn reward stream from the rollout — same
  length as assistant_spans.
- The terminal step's reward already captures the post-`submit_evidence_pack`
  jump because we use the per-step *delta* of the cumulative breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TrajectoryRewards:
    """Per-step rewards aligned to assistant turns of a single trajectory.

    `step_rewards[i]` is what the env returned at the i-th assistant turn.
    `parse_error_at[i]` flags whether the i-th turn was a parse error
    (used to bake in the format penalty separately if we ever want it
    detached from the env reward — currently the env doesn't penalize
    parse errors itself, we apply -0.05 here).
    """

    step_rewards: list[float]
    parse_error_at: list[bool]


FORMAT_PARSE_PENALTY: float = -0.05
"""Per-turn format penalty applied at parse-error positions.

Process Reward Models for LLM Agents (arxiv 2502.10325) recommends a
small negative reward floor for malformed tool calls — small enough not
to dominate the dense process reward, large enough to discourage format
breakage. -0.05 matches our cumulative dense budget (~1.0 / 40 turns).
"""


def build_per_token_rewards(
    assistant_spans: list[tuple[int, int]],
    step_rewards: list[float],
    parse_error_at: list[bool],
    seq_len: int,
    *,
    format_penalty: float = FORMAT_PARSE_PENALTY,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return a [seq_len] reward tensor with rewards at turn-end positions.

    For each assistant turn `i`:
      - reward goes at position `end - 1` (the last token of that turn)
      - if the turn had a parse error, ADD the format penalty on top of
        whatever the env returned for that step

    The env already applies most of the penalty (invalid_action_count
    increments, capped at -0.05 in `_penalties()`), so the format
    penalty here is additive on the reasoning-side: the model gets a
    direct token-level signal that the parse failed.

    Returns:
        [seq_len] float tensor, mostly zeros, non-zero at turn-end tokens.
    """
    assert len(assistant_spans) == len(step_rewards) == len(parse_error_at), (
        f"length mismatch: {len(assistant_spans)} spans, "
        f"{len(step_rewards)} rewards, {len(parse_error_at)} parse flags"
    )
    rewards = torch.zeros(seq_len, dtype=torch.float32, device=device)
    for (start, end), step_reward, was_parse_error in zip(assistant_spans, step_rewards, parse_error_at):
        if end <= start or end > seq_len:
            # Truncated turn — silently skip; the loss mask handles the rest.
            continue
        last_pos = end - 1
        rewards[last_pos] = float(step_reward)
        if was_parse_error:
            rewards[last_pos] = rewards[last_pos] + float(format_penalty)
    return rewards


def build_response_mask(
    assistant_spans: list[tuple[int, int]],
    seq_len: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """1 on assistant tokens, 0 elsewhere.

    Used both as the loss mask (only assistant tokens contribute to
    policy/value updates) and as the GAE whitening mask.
    """
    mask = torch.zeros(seq_len, dtype=torch.float32, device=device)
    for start, end in assistant_spans:
        if end <= start or end > seq_len:
            continue
        mask[start:end] = 1.0
    return mask


__all__ = [
    "FORMAT_PARSE_PENALTY",
    "TrajectoryRewards",
    "build_per_token_rewards",
    "build_response_mask",
]
