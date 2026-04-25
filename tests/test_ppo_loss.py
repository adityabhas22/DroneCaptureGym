"""Unit tests for PPO math primitives.

These run on CPU and don't require torch.cuda — they're a sanity check
that the GAE / KL / clipped-loss math we copied from SkyRL/verl behaves
correctly on small synthetic tensors before we point it at Qwen.
"""

from __future__ import annotations

import math

import pytest


torch = pytest.importorskip("torch")

from training.ppo.loss import (
    compute_approx_kl,
    compute_gae,
    gather_token_log_probs,
    masked_mean,
    masked_var,
    masked_whiten,
    ppo_critic_loss,
    ppo_policy_loss,
)


# ---------------------------------------------------------------------------
# masked_mean / masked_var / masked_whiten
# ---------------------------------------------------------------------------


def test_masked_mean_ignores_zeros():
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    assert masked_mean(values, mask).item() == pytest.approx(1.5)


def test_masked_mean_handles_fully_masked_row_without_nan():
    values = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.zeros_like(values)
    out = masked_mean(values, mask)
    assert torch.isfinite(out).all()
    assert out.item() == pytest.approx(0.0)


def test_masked_whiten_zero_mean_unit_var_over_mask():
    torch.manual_seed(0)
    values = torch.randn(2, 32)
    mask = torch.ones_like(values)
    mask[0, 16:] = 0.0  # mask out half of row 0
    whitened = masked_whiten(values, mask)
    # Mean over the masked positions should be ~0
    assert masked_mean(whitened, mask).abs().item() < 1e-5
    # Variance over the masked positions should be ~1
    assert masked_var(whitened, mask).item() == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# compute_gae
# ---------------------------------------------------------------------------


def test_gae_lambda_one_gamma_one_equals_monte_carlo_returns():
    # With λ=γ=1, GAE collapses to MC return minus baseline V(s).
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    values = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    mask = torch.ones_like(rewards)
    advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, lambd=1.0, whiten=False)
    # Returns should be MC: 1.0 at every position (terminal reward bootstraps back)
    assert torch.allclose(returns, torch.tensor([[1.0, 1.0, 1.0, 1.0]]), atol=1e-6)


def test_gae_zero_reward_zero_value_gives_zero_advantage():
    rewards = torch.zeros(1, 5)
    values = torch.zeros(1, 5)
    mask = torch.ones_like(rewards)
    advantages, returns = compute_gae(rewards, values, mask, gamma=0.99, lambd=0.95, whiten=False)
    assert torch.allclose(advantages, torch.zeros_like(advantages))
    assert torch.allclose(returns, torch.zeros_like(returns))


def test_gae_propagates_terminal_reward_with_discount():
    # Single trajectory, reward only at last step; lambda=1 so discount is purely gamma^t.
    rewards = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    values = torch.zeros_like(rewards)
    mask = torch.ones_like(rewards)
    gamma = 0.9
    advantages, _ = compute_gae(rewards, values, mask, gamma=gamma, lambd=1.0, whiten=False)
    # advantages[t] = gamma^(T-1-t) for last-step reward
    expected = torch.tensor([[gamma**3, gamma**2, gamma**1, gamma**0]])
    assert torch.allclose(advantages, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# compute_approx_kl
# ---------------------------------------------------------------------------


def test_kl_zero_when_distributions_match():
    log_p = torch.tensor([-1.0, -2.0, -0.5])
    kl = compute_approx_kl(log_p, log_p, estimator="k3")
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)


def test_kl_k3_strictly_nonnegative():
    torch.manual_seed(1)
    log_p = torch.randn(64) * 0.5
    log_q = torch.randn(64) * 0.5
    kl = compute_approx_kl(log_p, log_q, estimator="k3")
    # k3 is (ratio - 1) - log(ratio); algebraically ≥ 0 (with clamping).
    assert (kl > -1e-4).all(), f"k3 KL produced strongly negative entries: {kl.min()}"


def test_kl_mask_zeros_out_unmasked():
    log_p = torch.tensor([0.0, 0.0, 0.0, 0.0])
    log_q = torch.tensor([1.0, 1.0, 1.0, 1.0])
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    kl = compute_approx_kl(log_p, log_q, estimator="k3", mask=mask)
    assert kl[2:].abs().max().item() < 1e-9


# ---------------------------------------------------------------------------
# ppo_policy_loss
# ---------------------------------------------------------------------------


def test_ppo_policy_loss_at_old_logprobs_equals_negative_advantage():
    # ratio == 1 when log_probs == old_log_probs, so loss = -advantage * 1 = -advantage.
    log_probs = torch.tensor([[0.5, 0.5]])
    old = log_probs.clone()
    advantages = torch.tensor([[1.0, -2.0]])
    loss, metrics = ppo_policy_loss(log_probs, old, advantages, mask=torch.ones_like(advantages))
    # Mean loss = -(1.0 + -2.0) / 2 = 0.5
    assert loss.item() == pytest.approx(0.5)
    # Clip-frac is 0 when ratio is exactly 1
    assert metrics["clip_frac"] == pytest.approx(0.0)


def test_ppo_policy_loss_clips_high_ratio_with_positive_advantage():
    # Big positive advantage + ratio > 1 + eps_high → clipped to (1+eps_high).
    eps_high = 0.2
    log_probs = torch.tensor([[1.0]])
    old = torch.tensor([[0.0]])
    advantages = torch.tensor([[1.0]])
    loss, metrics = ppo_policy_loss(
        log_probs, old, advantages,
        eps_low=0.2, eps_high=eps_high,
        mask=torch.ones_like(advantages),
    )
    # ratio = exp(1.0) = 2.71... clipped to 1.2; surr1 = 2.71, surr2 = 1.2.
    # min(surr1, surr2) = 1.2 → loss = -1.2.
    assert loss.item() == pytest.approx(-(1.0 + eps_high), abs=1e-4)
    assert metrics["clip_frac"] > 0.5  # we clipped


# ---------------------------------------------------------------------------
# ppo_critic_loss
# ---------------------------------------------------------------------------


def test_critic_loss_is_zero_at_returns():
    values = torch.tensor([[1.0, 2.0, 3.0]])
    old_values = torch.tensor([[0.0, 0.0, 0.0]])  # different from values, but returns matches values
    returns = values.clone()
    loss, metrics = ppo_critic_loss(values, old_values, returns, clip=None, mask=torch.ones_like(values))
    assert loss.item() == pytest.approx(0.0)


def test_critic_loss_explained_variance_high_when_values_track_returns():
    torch.manual_seed(2)
    returns = torch.randn(4, 16)
    values = returns + 0.01 * torch.randn_like(returns)
    mask = torch.ones_like(returns)
    _, metrics = ppo_critic_loss(values, values, returns, clip=None, mask=mask)
    assert metrics["explained_variance"] > 0.95


# ---------------------------------------------------------------------------
# gather_token_log_probs
# ---------------------------------------------------------------------------


def test_gather_log_probs_shape_and_alignment():
    batch, seq, vocab = 2, 5, 7
    logits = torch.zeros(batch, seq, vocab)
    # Make the model maximally confident on token 3 at every position.
    logits[..., 3] = 100.0
    input_ids = torch.full((batch, seq), 3, dtype=torch.long)
    log_probs = gather_token_log_probs(logits, input_ids)
    assert log_probs.shape == (batch, seq - 1)
    # Confident model → log p ≈ 0 (after softmax on a near-one-hot).
    assert log_probs.abs().max().item() < 1e-3
