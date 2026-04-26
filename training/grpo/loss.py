"""GRPO loss: group-normalized advantages + clipped policy loss + KL-to-ref.

GRPO = Group Relative Policy Optimization (DeepSeek-R1, arXiv:2402.03300).

The key trick: instead of learning a value function, sample G rollouts
per prompt, compute the reward of each, and take group-normalized
advantage

    A_i = (r_i - mean({r_j})) / (std({r_j}) + eps)

Then apply the standard PPO clipped surrogate using these advantages:

    L = - E_t[ min(ratio_t * A, clip(ratio_t, 1-eps_low, 1+eps_high) * A) ]
        + kl_coef * KL(pi || pi_ref)
        - entropy_coef * H(pi)

Compared to PPO:

- No value head, no GAE, no critic loss.
- Advantages are constant across all tokens of a single rollout
  (broadcast across the assistant span).

We reuse ``compute_approx_kl``, ``masked_mean``, ``gather_token_log_probs``
and ``entropy_from_logits`` from the existing PPO ``loss`` module so we
do not duplicate the math.
"""

from __future__ import annotations

import torch

from training.ppo.loss import (  # re-exported for convenience
    compute_approx_kl,
    entropy_from_logits,
    gather_token_log_probs,
    masked_mean,
    masked_var,
)


def group_advantages(
    rewards: torch.Tensor,
    *,
    group_size: int,
    eps: float = 1.0e-4,
    clip: float | None = None,
) -> torch.Tensor:
    """Group-normalize a flat reward tensor.

    Args:
        rewards: shape ``[N * group_size]`` — rollout rewards laid out
            so that consecutive rollouts of the same prompt are
            adjacent (i.e. ``[p0_g0, p0_g1, ..., p0_gG, p1_g0, ...]``).
        group_size: G — number of rollouts per prompt.
        eps: numerical floor for the per-group std. Avoid divide-by-zero
            when every rollout in a group has the same reward.
        clip: optional symmetric clamp on the resulting advantage.
            Stops a single outlier reward from dominating the gradient.

    Returns:
        Tensor of shape ``[N * group_size]`` with mean ~ 0 and std ~ 1
        within each group.
    """

    if rewards.dim() != 1:
        raise ValueError(f"rewards must be 1D, got shape {tuple(rewards.shape)}")
    n_total = int(rewards.shape[0])
    if group_size <= 0 or n_total % group_size != 0:
        raise ValueError(
            f"rewards length {n_total} is not divisible by group_size {group_size}"
        )
    grouped = rewards.view(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    advantages = (grouped - mean) / (std + eps)
    if clip is not None and clip > 0:
        advantages = advantages.clamp(-clip, clip)
    return advantages.reshape(-1)


def grpo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages_per_seq: torch.Tensor,
    *,
    response_mask: torch.Tensor,
    eps_low: float = 0.2,
    eps_high: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Per-token PPO-clipped surrogate with broadcasted sequence advantages.

    Args:
        log_probs: ``[B, T-1]`` new policy per-token log-probs.
        old_log_probs: ``[B, T-1]`` rollout-time log-probs (no_grad).
        advantages_per_seq: ``[B]`` group-normalized advantage per
            rollout. Broadcast to per-token across the response mask.
        response_mask: ``[B, T-1]`` 1 on assistant tokens.
        eps_low / eps_high: PPO clip range. Asymmetric form supports
            DAPO-style relaxation; default symmetric 0.2/0.2.
    """

    if log_probs.shape != old_log_probs.shape:
        raise ValueError(
            f"log_probs {tuple(log_probs.shape)} != old_log_probs {tuple(old_log_probs.shape)}"
        )
    if response_mask.shape != log_probs.shape:
        raise ValueError(
            f"response_mask {tuple(response_mask.shape)} != log_probs {tuple(log_probs.shape)}"
        )

    log_ratio = (log_probs - old_log_probs).clamp(-20.0, 20.0)
    ratio = torch.exp(log_ratio)
    # Broadcast [B] -> [B, 1] -> [B, T-1] over the masked positions.
    advantages = advantages_per_seq.to(log_probs.dtype).unsqueeze(-1).expand_as(log_probs)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(1.0 - eps_low, 1.0 + eps_high) * advantages
    loss = -torch.min(surr1, surr2)
    final = masked_mean(loss, response_mask)

    with torch.no_grad():
        clip_frac = masked_mean((surr2 < surr1).float(), response_mask).item()
        approx_kl = masked_mean(-log_ratio, response_mask).item()
        ratio_mean = masked_mean(ratio, response_mask).item()

    return final, {
        "policy_loss": float(final.item()),
        "clip_frac": float(clip_frac),
        "ratio_mean": float(ratio_mean),
        "approx_kl": float(approx_kl),
    }


__all__ = [
    "compute_approx_kl",
    "entropy_from_logits",
    "gather_token_log_probs",
    "group_advantages",
    "grpo_policy_loss",
    "masked_mean",
    "masked_var",
]
