"""Unit tests for the GRPO loss math.

These are pure-PyTorch CPU tests — no model, no HF, no GPU. Validates:
- group_advantages produces zero-mean, unit-variance per group when
  std > eps; produces zero when all rewards in a group are equal.
- grpo_policy_loss sanity: ratio==1 => loss == -mean(advantage); ratio
  outside the clip range with a positive advantage uses the clipped
  surrogate.
"""

from __future__ import annotations

import math

import torch

from training.grpo.loss import (
    group_advantages,
    grpo_policy_loss,
    masked_mean,
)


def test_group_advantages_normalizes_within_group():
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 4.0])
    adv = group_advantages(rewards, group_size=3, eps=1e-6)
    # Two groups of 3
    g0, g1 = adv[:3], adv[3:]
    assert math.isclose(g0.mean().item(), 0.0, abs_tol=1e-6)
    assert math.isclose(g1.mean().item(), 0.0, abs_tol=1e-6)
    assert g0.std(unbiased=False).item() > 0.0
    assert g1.std(unbiased=False).item() > 0.0


def test_group_advantages_constant_group_returns_zero():
    rewards = torch.tensor([2.0, 2.0, 2.0, 2.0])
    adv = group_advantages(rewards, group_size=2, eps=1e-6)
    # Constant within each group → std==0 → advantage all zero (within eps).
    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-3)


def test_group_advantages_clip_caps_extremes():
    rewards = torch.tensor([0.0, 0.0, 100.0, 0.0])
    adv = group_advantages(rewards, group_size=4, eps=1e-6, clip=2.0)
    assert adv.max().item() <= 2.0 + 1e-6
    assert adv.min().item() >= -2.0 - 1e-6


def test_group_advantages_rejects_unaligned_lengths():
    rewards = torch.tensor([0.0, 1.0, 2.0])
    try:
        group_advantages(rewards, group_size=2)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("expected ValueError for unaligned rewards length")


def test_grpo_policy_loss_ratio_one_equals_neg_advantage():
    """When new == old logprobs, ratio==1, loss should equal -mean(advantage * mask)."""

    bsz, seq = 4, 8
    log_probs = torch.zeros(bsz, seq)
    old_log_probs = torch.zeros(bsz, seq)
    advantages = torch.tensor([1.0, -1.0, 2.0, -0.5])
    response_mask = torch.ones(bsz, seq)
    loss, metrics = grpo_policy_loss(
        log_probs, old_log_probs, advantages, response_mask=response_mask
    )
    expected = -float(advantages.mean().item())
    assert math.isclose(float(loss.item()), expected, abs_tol=1e-6)
    assert math.isclose(metrics["ratio_mean"], 1.0, abs_tol=1e-6)
    assert math.isclose(metrics["clip_frac"], 0.0, abs_tol=1e-6)


def test_grpo_policy_loss_clip_kicks_in_for_positive_advantage():
    """If new logprobs are much higher than old, ratio>1+eps high; with
    positive advantage the clipped surrogate dominates."""

    bsz, seq = 2, 4
    log_probs = torch.full((bsz, seq), 1.0)
    old_log_probs = torch.zeros(bsz, seq)
    advantages = torch.tensor([1.0, 1.0])
    response_mask = torch.ones(bsz, seq)
    loss, metrics = grpo_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        response_mask=response_mask,
        eps_low=0.2,
        eps_high=0.2,
    )
    # clip_frac should be 1 because every position has ratio>1+eps_high.
    assert metrics["clip_frac"] > 0.9
    # Loss with clipped surrogate at clip = (1+eps_high)*adv = 1.2 * 1 = 1.2
    # so loss = -mean(min(ratio*adv, 1.2*adv)). Both positive → min is 1.2.
    assert math.isclose(float(loss.item()), -1.2, abs_tol=1e-3)


def test_masked_mean_handles_partial_mask():
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    result = masked_mean(values, mask)
    # Sum of masked positions: 1 + 3 + 5 = 9, count = 3 → mean 3.0
    assert math.isclose(float(result.item()), 3.0, abs_tol=1e-6)
