"""PPO loss, GAE, and KL math.

Adapted (with attribution) from SkyRL:
    https://github.com/NovaSky-AI/SkyRL/blob/main/skyrl/backends/skyrl_train/utils/ppo_utils.py
which itself is adapted from verl:
    https://github.com/volcengine/verl/blob/main/verl/trainer/ppo/core_algos.py

Both upstream sources are Apache-2.0. The functions here are stripped to
the core math (no Ray actor registries, no off-policy correction, no
distributional variants — those would be added later if we need them).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Masked statistics
# ---------------------------------------------------------------------------


def masked_sum(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    return (values * mask).sum(dim=dim) if dim is not None else (values * mask).sum()


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Average `values` over the positions where `mask == 1`.

    Returns 0 for fully-masked rows rather than NaN — matters when a
    minibatch contains a trajectory that happens to have no assistant
    tokens (truncation edge case).
    """
    if dim is None:
        denom = mask.sum().clamp(min=1.0)
        return (values * mask).sum() / denom
    denom = mask.sum(dim=dim).clamp(min=1.0)
    return (values * mask).sum(dim=dim) / denom


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    mean = masked_mean(values, mask)
    centered = values - mean
    var = masked_mean(centered.square(), mask)
    if unbiased:
        n = mask.sum()
        if n > 1:
            var = var * (n / (n - 1))
    return var


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Standardize values using only masked positions for the moments."""
    mean = masked_mean(values, mask)
    var = masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened


# ---------------------------------------------------------------------------
# KL — Schulman approximations http://joschu.net/blog/kl-approx.html
# ---------------------------------------------------------------------------


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    *,
    estimator: str = "k3",
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token KL(π || π_ref).

    `k3` is the unbiased, low-variance, strictly-positive estimator we
    default to — same as verl's `low_var_kl`. It's what stops noisy
    per-token KL from poisoning advantage estimates.
    """
    if estimator == "k1":
        kld = log_probs - log_probs_ref
    elif estimator == "abs":
        kld = (log_probs - log_probs_ref).abs()
    elif estimator == "k2":
        kld = 0.5 * (log_probs - log_probs_ref).square()
    elif estimator == "k3":
        delta = (log_probs_ref - log_probs).clamp(-20.0, 20.0)
        ratio = torch.exp(delta)
        kld = (ratio - delta - 1).contiguous().clamp(-10.0, 10.0)
    else:
        raise ValueError(f"unknown KL estimator: {estimator!r}")
    if mask is not None:
        kld = kld * mask
    return kld


# ---------------------------------------------------------------------------
# Generalized advantage estimation
# ---------------------------------------------------------------------------


def compute_gae(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gamma: float = 0.99,
    lambd: float = 0.95,
    whiten: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-level GAE with whitening.

    Args:
        token_level_rewards: [B, T] — non-zero only at turn-end tokens
            (placed by `reward_placement.build_per_token_rewards`).
        values: [B, T] — V(s_t) from the critic at every position.
        response_mask: [B, T] — 1 on assistant tokens (where loss applies),
            0 elsewhere. Used both for whitening and as a hint that
            preceding tokens get advantages via value bootstrapping.
        gamma: discount. 0.99 for 40-step horizons (1.0 = MC, high variance).
        lambd: GAE λ. 0.95 = standard PPO default; lower trades bias for
            variance.
        whiten: standardize advantages over the masked positions.

    Returns:
        (advantages, returns) both [B, T].
    """
    with torch.no_grad():
        gen_len = token_level_rewards.shape[-1]
        lastgaelam = torch.zeros_like(token_level_rewards[:, 0])
        advantages_rev: list[torch.Tensor] = []
        for t in reversed(range(gen_len)):
            next_v = values[:, t + 1] if t < gen_len - 1 else torch.zeros_like(values[:, 0])
            delta = token_level_rewards[:, t] + gamma * next_v - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_rev.append(lastgaelam)
        advantages = torch.stack(list(reversed(advantages_rev)), dim=1)
        returns = advantages + values
        if whiten:
            advantages = masked_whiten(advantages, response_mask)
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO clipped policy loss
# ---------------------------------------------------------------------------


def ppo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
    mask: torch.Tensor | None = None,
    dual_clip_c: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Clipped PPO policy loss with optional dual-clip.

    Asymmetric eps_low / eps_high follows DAPO (https://arxiv.org/abs/2503.14476)
    — a small high-side relaxation (0.28 in DAPO) helps when negative
    advantages would otherwise pull the ratio down too far. We keep
    symmetric 0.2/0.2 by default.
    """
    log_ratio = (log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * advantages
    loss = -torch.min(surr1, surr2)

    if dual_clip_c is not None and dual_clip_c > 1.0:
        # Dual-clip from "Mastering Complex Control" (Ye et al. 2020):
        # cap the magnitude of negative-advantage updates, useful when
        # a few outlier completions blow up the gradient.
        clip_neg = -advantages * dual_clip_c
        loss_neg = torch.min(clip_neg, loss)
        loss = torch.where(advantages < 0, loss_neg, loss)

    if mask is None:
        mask = torch.ones_like(loss)
    final_loss = masked_mean(loss, mask)

    with torch.no_grad():
        clip_frac = masked_mean((surr2 < surr1).float(), mask).item()
        approx_kl = masked_mean(-log_ratio, mask).item()  # ratio-based proxy
        ratio_mean = masked_mean(ratio, mask).item()

    return final_loss, {
        "policy_loss": final_loss.item(),
        "clip_frac": clip_frac,
        "ratio_mean": ratio_mean,
        "approx_kl": approx_kl,
    }


# ---------------------------------------------------------------------------
# Critic / value loss with optional clipping
# ---------------------------------------------------------------------------


def ppo_critic_loss(
    values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip: float | None = 0.2,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Squared-error value loss with optional value clipping (PPO-style).

    Clipping the value update to ±`clip` around the rollout's `old_values`
    prevents the critic from chasing noise within a single PPO update.
    """
    if mask is None:
        mask = torch.ones_like(values)
    if clip is not None and clip > 0:
        v_clipped = old_values + (values - old_values).clamp(-clip, clip)
        loss = torch.maximum((values - returns).square(), (v_clipped - returns).square())
    else:
        loss = (values - returns).square()
    final = 0.5 * masked_mean(loss, mask)
    with torch.no_grad():
        explained_var = 1.0 - (
            masked_var(returns - values, mask) / (masked_var(returns, mask) + 1e-8)
        )
    return final, {
        "value_loss": final.item(),
        "explained_variance": float(explained_var.item()),
    }


# ---------------------------------------------------------------------------
# Per-token log-probs from logits + ids
# ---------------------------------------------------------------------------


def gather_token_log_probs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute log p(input_ids[:, t] | logits[:, t-1]).

    Returns shape [B, T-1]. Caller is responsible for shifting / masking
    to align with the appropriate response positions.
    """
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:].unsqueeze(-1)
    return log_probs.gather(-1, targets).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy of the categorical distribution over the vocab."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


__all__ = [
    "compute_approx_kl",
    "compute_gae",
    "entropy_from_logits",
    "gather_token_log_probs",
    "masked_mean",
    "masked_sum",
    "masked_var",
    "masked_whiten",
    "ppo_critic_loss",
    "ppo_policy_loss",
]
