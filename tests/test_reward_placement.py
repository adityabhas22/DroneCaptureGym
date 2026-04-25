"""Tests for token-level reward placement at assistant turn boundaries."""

from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from training.ppo.reward_placement import (
    FORMAT_PARSE_PENALTY,
    build_per_token_rewards,
    build_response_mask,
)


def test_single_turn_reward_lands_on_last_token():
    # Turn occupies positions [3, 7); reward should land at position 6.
    rewards = build_per_token_rewards(
        assistant_spans=[(3, 7)],
        step_rewards=[0.42],
        parse_error_at=[False],
        seq_len=10,
    )
    assert rewards.shape == (10,)
    assert rewards[6].item() == pytest.approx(0.42)
    # Everything else should be zero.
    rewards[6] = 0.0
    assert rewards.abs().max().item() == 0.0


def test_multi_turn_rewards_independent_per_turn():
    rewards = build_per_token_rewards(
        assistant_spans=[(2, 4), (5, 9), (10, 14)],
        step_rewards=[0.1, 0.2, -0.3],
        parse_error_at=[False, False, False],
        seq_len=15,
    )
    assert rewards[3].item() == pytest.approx(0.1)
    assert rewards[8].item() == pytest.approx(0.2)
    assert rewards[13].item() == pytest.approx(-0.3)
    # Other positions zero.
    nonzero_positions = (rewards != 0).nonzero(as_tuple=True)[0].tolist()
    assert nonzero_positions == [3, 8, 13]


def test_parse_error_adds_format_penalty():
    rewards = build_per_token_rewards(
        assistant_spans=[(0, 3), (4, 8)],
        step_rewards=[0.0, 0.1],
        parse_error_at=[True, False],
        seq_len=10,
    )
    # Turn 0: env reward 0.0 + format penalty -0.05 = -0.05 at position 2.
    assert rewards[2].item() == pytest.approx(FORMAT_PARSE_PENALTY)
    # Turn 1: env reward 0.1, no penalty, at position 7.
    assert rewards[7].item() == pytest.approx(0.1)


def test_truncated_span_is_silently_skipped():
    # The third turn's `end` exceeds seq_len; should be dropped without error.
    rewards = build_per_token_rewards(
        assistant_spans=[(0, 3), (4, 8), (9, 20)],
        step_rewards=[0.1, 0.2, 0.3],
        parse_error_at=[False, False, False],
        seq_len=10,
    )
    # Only first two turns landed.
    assert rewards[2].item() == pytest.approx(0.1)
    assert rewards[7].item() == pytest.approx(0.2)
    # Beyond seq_len, no reward written; positions 8-9 should remain 0.
    assert rewards[8].item() == 0.0
    assert rewards[9].item() == 0.0


def test_response_mask_marks_only_assistant_tokens():
    mask = build_response_mask(
        assistant_spans=[(2, 4), (6, 9)],
        seq_len=12,
    )
    assert mask.shape == (12,)
    expected = torch.zeros(12)
    expected[2:4] = 1.0
    expected[6:9] = 1.0
    assert torch.equal(mask, expected)


def test_length_mismatch_raises_assertion():
    with pytest.raises(AssertionError):
        build_per_token_rewards(
            assistant_spans=[(0, 3)],
            step_rewards=[0.1, 0.2],         # length 2 vs 1 span
            parse_error_at=[False, False],   # length 2 vs 1 span
            seq_len=10,
        )
