"""Single-process PPO trainer for DroneCaptureOps.

Architecture:
    - 1 GPU (H200 by default), 1 process, no Ray.
    - Policy: Qwen base + LoRA adapter (continued from SFT warmstart).
    - Value head: fresh `nn.Linear(hidden_size, 1)` zero-initialized,
      forwarded on top of the policy's last hidden states.
    - Reference: SAME backbone with adapter disabled via PEFT's
      `disable_adapter()` context — saves us a second model copy.
    - Rollouts: shared vLLM engine with LoRA hot-swap each PPO step
      (save adapter → new LoRARequest → next rollout batch).
    - Forward: training model in eval mode for old_log_probs/values
      and ref_log_probs, then train mode for K PPO update epochs.

The math (`compute_gae`, `ppo_policy_loss`, `ppo_critic_loss`,
`compute_approx_kl`) lives in `loss.py` and is adapted from
SkyRL/verl with attribution.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dronecaptureops.agent.hf_local_engine import HFLocalEngine
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS
from training.ppo.config import (
    AlgorithmConfig,
    CriticWarmupConfig,
    EvalConfig,
    LoRAConfig,
    OptimizerConfig,
    PPOTrainConfig,
    RolloutConfig,
)
from training.ppo.loss import (
    compute_approx_kl,
    compute_gae,
    entropy_from_logits,
    gather_token_log_probs,
    masked_mean,
    ppo_critic_loss,
    ppo_policy_loss,
)
from training.ppo.reward_placement import (
    FORMAT_PARSE_PENALTY,
    build_per_token_rewards,
)
from training.ppo.rollout_pool import (
    PPORolloutOutput,
    PPORolloutSpec,
    run_rollout_batch,
)
from training.ppo.tokenization import (
    TokenizedTrajectory,
    pad_trajectories,
    tokenize_trajectory,
)


LOG = logging.getLogger("dronecaptureops.ppo.trainer")


# ---------------------------------------------------------------------------
# Actor-critic model
# ---------------------------------------------------------------------------


class ActorCritic(nn.Module):
    """Policy with a fresh value head on top of the LoRA-adapted backbone.

    Reference forward: enter `model.policy.disable_adapter()` to get the
    same backbone with LoRA disabled. The value head is NOT trained
    during reference forwards — caller should `.detach()` its output.
    """

    def __init__(self, policy: nn.Module, hidden_size: int):
        super().__init__()
        self.policy = policy
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        nn.init.zeros_(self.value_head.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        compute_values: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        outputs = self.policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=compute_values,
            use_cache=False,
        )
        values: torch.Tensor | None = None
        if compute_values:
            hidden = outputs.hidden_states[-1]
            values = self.value_head(hidden.to(self.value_head.weight.dtype)).squeeze(-1)
        return outputs.logits, values

    def trainable_parameter_groups(self, actor_lr: float, critic_lr: float) -> list[dict[str, Any]]:
        """Param groups for AdamW: LoRA adapters at actor_lr, value head at critic_lr."""
        actor_params: list[nn.Parameter] = []
        critic_params: list[nn.Parameter] = list(self.value_head.parameters())
        for name, p in self.policy.named_parameters():
            if p.requires_grad:
                actor_params.append(p)
        return [
            {"params": actor_params, "lr": actor_lr, "name": "actor_lora"},
            {"params": critic_params, "lr": critic_lr, "name": "critic_value_head"},
        ]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    step: int
    rollout_secs: float
    forward_secs: float
    update_secs: float
    n_rollouts: int
    n_assistant_turns: int
    mean_episode_steps: float
    mean_total_reward: float
    mean_terminal_reward: float
    success_rate: float
    parse_error_rate: float
    policy_loss: float = 0.0
    value_loss: float = 0.0
    explained_variance: float = 0.0
    clip_frac: float = 0.0
    approx_kl: float = 0.0
    kl_to_ref: float = 0.0
    entropy: float = 0.0
    ratio_mean: float = 1.0
    grad_norm: float = 0.0


class PPOTrainer:
    def __init__(self, config: PPOTrainConfig):
        self.cfg = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_cache = Path(config.vllm_lora_dir)
        self.lora_cache.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"

        # Lazy heavyweight imports — keeps tests fast on CPU-only hosts.
        import torch as _torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Pinned references so we can use them later without re-importing.
        self._torch = _torch
        self._PeftModel = PeftModel
        self._get_peft_model = get_peft_model
        self._LoraConfig = LoraConfig

        self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        self.dtype = _torch.bfloat16 if self.device.type == "cuda" else _torch.float32

        LOG.info("loading tokenizer: %s", config.tokenizer_name or config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        LOG.info("loading base model: %s (%s)", config.model_name, self.dtype)
        base = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=self.dtype,
        )
        base.gradient_checkpointing_enable()

        # Continue from SFT LoRA, or attach a fresh adapter.
        # `sft_checkpoint` can be either a local path OR a Hub repo_id.
        # Subfolder syntax: "repo_id:subfolder" (e.g.,
        # "adityabhaskara/dronecaptureops-sft-qwen3-4b:last-checkpoint").
        sft_spec = config.sft_checkpoint
        sft_loaded = False
        if sft_spec:
            if ":" in sft_spec and not Path(sft_spec).exists():
                repo_id, subfolder = sft_spec.split(":", 1)
                load_target, load_kwargs = repo_id, {"subfolder": subfolder}
            else:
                local_path = Path(sft_spec)
                if local_path.exists():
                    load_target, load_kwargs = str(local_path), {}
                else:
                    # Treat as bare Hub repo_id (no subfolder).
                    load_target, load_kwargs = sft_spec, {}
            try:
                LOG.info("loading SFT LoRA: %s (kwargs=%s)", load_target, load_kwargs)
                policy = PeftModel.from_pretrained(base, load_target, is_trainable=True, **load_kwargs)
                sft_loaded = True
            except Exception as exc:
                LOG.warning(
                    "SFT checkpoint %s could not be loaded (%s); falling back to fresh LoRA.",
                    sft_spec, exc,
                )
        if not sft_loaded:
            if sft_spec is not None:
                LOG.warning(
                    "SFT checkpoint not loaded — attaching a fresh LoRA. "
                    "Set --sft-checkpoint to a valid local path or Hub repo_id "
                    "(optionally with :subfolder) to continue from a real warmstart.",
                )
            LOG.info("attaching fresh LoRA: r=%d alpha=%d", config.lora.rank, config.lora.alpha)
            lora_cfg = LoraConfig(
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=list(config.lora.target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            policy = get_peft_model(base, lora_cfg)

        hidden_size = base.config.hidden_size
        self.model = ActorCritic(policy=policy, hidden_size=hidden_size).to(self.device)

        # Make sure LoRA params and value head are the only trainable surface.
        for name, p in self.model.named_parameters():
            if "lora_" in name or name.startswith("value_head"):
                p.requires_grad = True
            else:
                p.requires_grad = False

        # Optimizer
        from torch.optim import AdamW

        param_groups = self.model.trainable_parameter_groups(
            actor_lr=config.optimizer.actor_lr,
            critic_lr=config.optimizer.critic_lr,
        )
        self.optimizer = AdamW(
            param_groups,
            betas=tuple(config.optimizer.betas),
            weight_decay=config.optimizer.weight_decay,
        )

        # In-process HF rollout engine (replaces vLLM — see hf_local_engine.py
        # for the why). Initialized lazily so unit tests don't pay boot cost.
        self._engine: HFLocalEngine | None = None
        self._lora_request_step: int = 0
        self._sft_lora_path: Path | None = None

        # Wandb (optional)
        self._wandb = None
        if config.wandb_mode != "disabled":
            try:
                import wandb

                self._wandb = wandb
                self._wandb.init(
                    project=config.wandb_project or "dronecaptureops-ppo",
                    name=config.wandb_run_name,
                    mode=config.wandb_mode,
                    config=config.model_dump(),
                )
            except ImportError:
                LOG.warning("wandb not installed; logging to JSONL only")
                self._wandb = None

    # ---------- Rollout engine lifecycle ----------

    def _ensure_engine(self) -> HFLocalEngine:
        """Lazily wrap our trainable model in the HFLocalEngine batcher.

        We pass the live PEFT-wrapped policy (`self.model.policy`), so the
        engine sees the current LoRA weights for free — no save/reload, no
        IPC, no separate vLLM process.
        """

        if self._engine is None:
            LOG.info(
                "starting HFLocalEngine (max_batch=%d, max_workers=%d)",
                self.cfg.rollout.rollout_batch_size,
                self.cfg.rollout.max_workers,
            )
            self._engine = HFLocalEngine(
                model=self.model.policy,
                tokenizer=self.tokenizer,
                device=self.device,
                max_batch_size=self.cfg.rollout.rollout_batch_size,
                max_history_steps=self.cfg.rollout.max_history_steps,
                enable_thinking=False,
            )
        return self._engine

    def _build_lora_request(self, step_id: int):
        """No-op stub kept for call-site compatibility.

        The legacy vLLM path needed a per-step LoRARequest so vLLM could
        hot-swap to the freshly-saved adapter. With HFLocalEngine we use
        the trainer's live model directly, so there's nothing to swap.
        Returns None so any downstream code that passes this through
        gets the "no adapter override" sentinel.
        """

        return None

    # ---------- Task sampling ----------

    def _resolved_train_tasks(self) -> list[str]:
        if self.cfg.train_tasks:
            return list(self.cfg.train_tasks)
        held_out = set(self.cfg.eval.held_out_tasks)
        all_ids = [task_id for task_id in SOLAR_TASKS if task_id not in held_out]
        if not all_ids:
            raise RuntimeError("no training tasks resolved — every task is held out")
        return all_ids

    def _sample_specs(self, rng: random.Random) -> list[PPORolloutSpec]:
        tasks = self._resolved_train_tasks()
        n = self.cfg.rollout.rollout_batch_size
        return [
            PPORolloutSpec(
                task_id=rng.choice(tasks),
                seed=rng.randint(0, 2**31 - 1),
            )
            for _ in range(n)
        ]

    # ---------- Forward pass helpers ----------

    def _forward_logprobs_values(
        self,
        batch: dict[str, torch.Tensor],
        *,
        compute_values: bool,
        train_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """One forward pass; returns per-token log-probs + values + entropy.

        log-probs and entropy are aligned with `input_ids[:, 1:]` (i.e.
        position t corresponds to predicting input_ids[:, t+1]). Caller
        is responsible for shifting masks accordingly.
        """
        was_training = self.model.training
        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        try:
            ctx = torch.enable_grad() if train_mode else torch.no_grad()
            with ctx:
                logits, values = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    compute_values=compute_values,
                )
                # Per-token log-prob of input_ids[t] given logits[t-1]
                logprobs = gather_token_log_probs(logits, batch["input_ids"])
                # Entropy at each input position (exclude final token's logits)
                entropy = entropy_from_logits(logits[:, :-1, :])
        finally:
            self.model.train(was_training)
        # values is [B, T]; align to logprobs ([B, T-1]) by dropping last position.
        values_aligned = None if values is None else values[:, :-1]
        return logprobs, values_aligned, entropy

    def _forward_ref_logprobs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with the LoRA adapter disabled — gives π_ref."""
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                with self.model.policy.disable_adapter():
                    logits, _ = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        compute_values=False,
                    )
                    return gather_token_log_probs(logits, batch["input_ids"])
        finally:
            self.model.train(was_training)

    # ---------- Reward placement ----------

    def _build_step_rewards(self, output: PPORolloutOutput) -> tuple[list[float], list[bool]]:
        """Per-turn rewards (delta of cumulative env reward) and parse-error flags."""
        traj = output.result.trajectory
        prev_total = 0.0
        rewards: list[float] = []
        parse_flags: list[bool] = []
        for step in traj:
            delta = float(step.reward) - prev_total
            rewards.append(delta)
            parse_flags.append(step.parse_error is not None)
            prev_total = float(step.reward)
        return rewards, parse_flags

    # ---------- Main training step ----------

    def step(self, step_idx: int, rng: random.Random) -> StepMetrics:
        cfg = self.cfg
        torch_ = self._torch

        # 1. Save current LoRA, build LoRARequest, run rollouts.
        lora_request = self._build_lora_request(step_idx)
        engine = self._ensure_engine()
        specs = self._sample_specs(rng)

        t0 = time.perf_counter()
        outputs = run_rollout_batch(
            specs,
            engine=engine,
            lora_request=lora_request,
            max_workers=cfg.rollout.max_workers,
            temperature=cfg.rollout.temperature,
            top_p=cfg.rollout.top_p,
            max_tokens=cfg.rollout.max_new_tokens_per_turn,
            max_history_steps=cfg.rollout.max_history_steps,
            max_steps=cfg.rollout.max_episode_steps,
        )
        t_rollout = time.perf_counter() - t0

        # Free generation-time KV cache before the training forward/backward
        # — engine and trainer share the same CUDA caching allocator, so without
        # this the rollout's KV cache stays reserved and the backward pass OOMs.
        if self.device.type == "cuda":
            self._torch.cuda.empty_cache()

        # 2. Tokenize trajectories.
        tokenized: list[TokenizedTrajectory] = []
        per_traj_step_rewards: list[list[float]] = []
        per_traj_parse_flags: list[list[bool]] = []
        kept_outputs: list[PPORolloutOutput] = []
        for output in outputs:
            traj = tokenize_trajectory(
                output.messages,
                tokenizer=self.tokenizer,
                max_length=cfg.max_total_length,
            )
            if traj.n_assistant_turns == 0:
                continue
            rewards, parse_flags = self._build_step_rewards(output)
            # Truncate rewards/parse_flags to the surviving turn count.
            rewards = rewards[: traj.n_assistant_turns]
            parse_flags = parse_flags[: traj.n_assistant_turns]
            tokenized.append(traj)
            per_traj_step_rewards.append(rewards)
            per_traj_parse_flags.append(parse_flags)
            kept_outputs.append(output)

        if not tokenized:
            LOG.warning("step %d produced 0 trainable trajectories; skipping update", step_idx)
            return self._summarize_metrics(step_idx, outputs, kept_outputs, t_rollout, 0.0, 0.0)

        batch = pad_trajectories(tokenized, pad_token_id=self.tokenizer.pad_token_id, device=self.device)
        seq_len = int(batch["input_ids"].shape[1])
        bsz = int(batch["input_ids"].shape[0])

        # Build per-token rewards (B, T) using the assistant spans of each trajectory.
        per_token_rewards = torch_.zeros((bsz, seq_len), dtype=torch_.float32, device=self.device)
        for b, (traj, rewards, parse_flags) in enumerate(zip(tokenized, per_traj_step_rewards, per_traj_parse_flags)):
            row = build_per_token_rewards(
                assistant_spans=traj.assistant_spans,
                step_rewards=rewards,
                parse_error_at=parse_flags,
                seq_len=seq_len,
                format_penalty=FORMAT_PARSE_PENALTY,
                device=self.device,
            )
            per_token_rewards[b] = row

        # 3. Forward pass for old logprobs/values + ref logprobs.
        t0 = time.perf_counter()
        old_logprobs, old_values, _ = self._forward_logprobs_values(
            batch, compute_values=True, train_mode=False
        )
        ref_logprobs = self._forward_ref_logprobs(batch)
        t_forward = time.perf_counter() - t0

        # All log-probs are at positions [1, T) corresponding to input_ids[1:].
        # Shift the reward tensor and response_mask by 1 to align: token at
        # logprob index t corresponds to input_ids position t+1. Reward at
        # position p in reward tensor → align to logprob index p-1.
        rewards_shifted = per_token_rewards[:, 1:]
        response_mask_shifted = batch["response_mask"][:, 1:]

        # 4. GAE + KL-as-reward (we put KL in the loss, so reward stays as-is).
        advantages, returns = compute_gae(
            rewards_shifted,
            old_values,
            response_mask_shifted,
            gamma=cfg.algorithm.gamma,
            lambd=cfg.algorithm.lam,
            whiten=cfg.algorithm.whiten_advantages,
        )

        # 5. K-epoch PPO update.
        t0 = time.perf_counter()
        all_indices = list(range(bsz))
        update_metrics: list[dict[str, float]] = []
        for _epoch in range(cfg.ppo_epochs):
            rng.shuffle(all_indices)
            for start in range(0, bsz, cfg.minibatch_size):
                idx = all_indices[start : start + cfg.minibatch_size]
                if not idx:
                    continue
                # Micro-batched gradient accumulation within the minibatch.
                self.optimizer.zero_grad(set_to_none=True)
                accum_metrics: list[dict[str, float]] = []
                for micro_start in range(0, len(idx), cfg.micro_batch_size):
                    micro = idx[micro_start : micro_start + cfg.micro_batch_size]
                    micro_batch = {k: v[micro] for k, v in batch.items()}
                    new_logprobs, new_values, new_entropy = self._forward_logprobs_values(
                        micro_batch, compute_values=True, train_mode=True
                    )
                    micro_response_mask = micro_batch["response_mask"][:, 1:]
                    micro_old_logprobs = old_logprobs[micro]
                    micro_old_values = old_values[micro]
                    micro_ref_logprobs = ref_logprobs[micro]
                    micro_advantages = advantages[micro]
                    micro_returns = returns[micro]

                    policy_loss, p_metrics = ppo_policy_loss(
                        new_logprobs,
                        micro_old_logprobs,
                        micro_advantages,
                        eps_low=cfg.algorithm.eps_clip_low,
                        eps_high=cfg.algorithm.eps_clip_high,
                        mask=micro_response_mask,
                        dual_clip_c=cfg.algorithm.dual_clip_c,
                    )
                    value_loss, v_metrics = ppo_critic_loss(
                        new_values,
                        micro_old_values,
                        micro_returns,
                        clip=cfg.algorithm.value_clip,
                        mask=micro_response_mask,
                    )
                    kl = compute_approx_kl(
                        new_logprobs,
                        micro_ref_logprobs,
                        estimator=cfg.algorithm.kl_estimator,
                        mask=micro_response_mask,
                    )
                    kl_term = masked_mean(kl, micro_response_mask)
                    entropy_term = masked_mean(new_entropy, micro_response_mask)

                    loss = (
                        policy_loss
                        + 0.5 * value_loss
                        + cfg.algorithm.kl_coef * kl_term
                        - cfg.algorithm.entropy_coef * entropy_term
                    )
                    # Scale for accumulation across micro-batches in this minibatch.
                    accum_scale = float(len(micro)) / float(len(idx))
                    (loss * accum_scale).backward()
                    accum_metrics.append({
                        **p_metrics,
                        **v_metrics,
                        "kl_to_ref": float(kl_term.item()),
                        "entropy": float(entropy_term.item()),
                    })

                grad_norm = torch_.nn.utils.clip_grad_norm_(
                    [p for g in self.optimizer.param_groups for p in g["params"]],
                    cfg.optimizer.max_grad_norm,
                )
                self.optimizer.step()
                update_metrics.append({**_avg(accum_metrics), "grad_norm": float(grad_norm)})
        t_update = time.perf_counter() - t0

        metrics = self._summarize_metrics(step_idx, outputs, kept_outputs, t_rollout, t_forward, t_update)
        if update_metrics:
            avg = _avg(update_metrics)
            metrics.policy_loss = avg.get("policy_loss", 0.0)
            metrics.value_loss = avg.get("value_loss", 0.0)
            metrics.explained_variance = avg.get("explained_variance", 0.0)
            metrics.clip_frac = avg.get("clip_frac", 0.0)
            metrics.approx_kl = avg.get("approx_kl", 0.0)
            metrics.kl_to_ref = avg.get("kl_to_ref", 0.0)
            metrics.entropy = avg.get("entropy", 0.0)
            metrics.ratio_mean = avg.get("ratio_mean", 1.0)
            metrics.grad_norm = avg.get("grad_norm", 0.0)
        return metrics

    # ---------- Aggregation ----------

    def _summarize_metrics(
        self,
        step_idx: int,
        all_outputs: list[PPORolloutOutput],
        kept_outputs: list[PPORolloutOutput],
        t_rollout: float,
        t_forward: float,
        t_update: float,
    ) -> StepMetrics:
        n = len(all_outputs)
        if n == 0:
            return StepMetrics(
                step=step_idx, rollout_secs=t_rollout, forward_secs=t_forward, update_secs=t_update,
                n_rollouts=0, n_assistant_turns=0, mean_episode_steps=0.0, mean_total_reward=0.0,
                mean_terminal_reward=0.0, success_rate=0.0, parse_error_rate=0.0,
            )
        n_turns = sum(len(o.result.trajectory) for o in kept_outputs)
        success = sum(1 for o in all_outputs if o.result.success) / n
        mean_steps = sum(o.result.steps for o in all_outputs) / n
        mean_reward = sum(o.result.total_reward for o in all_outputs) / n
        # Terminal reward = total at the last step (which == mean_reward for these envs)
        mean_terminal = mean_reward
        parse_errors = sum(
            1 for o in all_outputs for s in o.result.trajectory if s.parse_error is not None
        )
        total_steps = sum(len(o.result.trajectory) for o in all_outputs) or 1
        return StepMetrics(
            step=step_idx,
            rollout_secs=t_rollout,
            forward_secs=t_forward,
            update_secs=t_update,
            n_rollouts=n,
            n_assistant_turns=n_turns,
            mean_episode_steps=float(mean_steps),
            mean_total_reward=float(mean_reward),
            mean_terminal_reward=float(mean_terminal),
            success_rate=float(success),
            parse_error_rate=float(parse_errors / total_steps),
        )

    # ---------- Persistence ----------

    def save_checkpoint(self, step: int, *, label: str = "step") -> Path:
        ckpt_dir = self.output_dir / f"{label}_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # LoRA adapter
        self.model.policy.save_pretrained(str(ckpt_dir / "adapter"))
        # Value head
        self._torch.save(self.model.value_head.state_dict(), ckpt_dir / "value_head.pt")
        # Tokenizer (so eval/inference don't have to look it up again)
        self.tokenizer.save_pretrained(str(ckpt_dir / "tokenizer"))
        # Trainer config snapshot
        (ckpt_dir / "config.json").write_text(json.dumps(self.cfg.model_dump(), indent=2, sort_keys=True))
        LOG.info("checkpoint saved: %s", ckpt_dir)
        return ckpt_dir

    def log_metrics(self, metrics: StepMetrics) -> None:
        record = metrics.__dict__
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        if self._wandb is not None:
            self._wandb.log(record, step=metrics.step)
        LOG.info(
            "step %d | reward=%.3f success=%.2f parse_err=%.2f kl=%.3f ent=%.3f vloss=%.3f ev=%.2f t_roll=%.1fs t_upd=%.1fs",
            metrics.step,
            metrics.mean_total_reward,
            metrics.success_rate,
            metrics.parse_error_rate,
            metrics.kl_to_ref,
            metrics.entropy,
            metrics.value_loss,
            metrics.explained_variance,
            metrics.rollout_secs,
            metrics.update_secs,
        )

    # ---------- Top-level fit() ----------

    def fit(self) -> None:
        cfg = self.cfg
        rng = random.Random(cfg.seed)

        # Ensure we're saving the initial (SFT-equivalent) adapter once so
        # vLLM can be started before the first PPO step.
        self._build_lora_request(step_id=0)

        # Optional critic warmup — keeps actor frozen, trains value head only.
        if cfg.critic_warmup.enabled and cfg.critic_warmup.steps > 0:
            self._run_critic_warmup(rng)

        for step in range(1, cfg.total_steps + 1):
            metrics = self.step(step, rng)
            self.log_metrics(metrics)
            if step % cfg.save_interval_steps == 0 or step == cfg.total_steps:
                self.save_checkpoint(step)

        # Final
        self.save_checkpoint(cfg.total_steps, label="final")
        if self._wandb is not None:
            self._wandb.finish()

    def _run_critic_warmup(self, rng: random.Random) -> None:
        """Train the value head on rollouts from the SFT policy.

        Freezes the actor (LoRA params) and trains only the value head
        to predict Monte-Carlo returns. Keeps the critic from giving
        garbage advantages during the first ~1k PPO steps.
        """
        cfg = self.cfg
        warmup_steps = cfg.critic_warmup.steps
        LOG.info("critic warmup: %d steps", warmup_steps)
        # Freeze actor
        for p in self.model.policy.parameters():
            p.requires_grad = False
        warmup_lr = cfg.optimizer.actor_lr * cfg.critic_warmup.lr_multiplier
        for group in self.optimizer.param_groups:
            if group.get("name") == "critic_value_head":
                group["lr"] = warmup_lr

        for w in range(1, warmup_steps + 1):
            t_step = time.perf_counter()
            LOG.info("critic warmup %d/%d: starting rollouts (n=%d)", w, warmup_steps, cfg.rollout.rollout_batch_size)
            specs = self._sample_specs(rng)
            engine = self._ensure_engine()
            lora_request = self._build_lora_request(step_id=0)  # SFT adapter
            t_roll = time.perf_counter()
            outputs = run_rollout_batch(
                specs,
                engine=engine,
                lora_request=lora_request,
                max_workers=cfg.rollout.max_workers,
                temperature=cfg.rollout.temperature,
                top_p=cfg.rollout.top_p,
                max_tokens=cfg.rollout.max_new_tokens_per_turn,
                max_history_steps=cfg.rollout.max_history_steps,
                max_steps=cfg.rollout.max_episode_steps,
            )
            t_roll = time.perf_counter() - t_roll
            mem_after_roll = self._torch.cuda.memory_allocated() / 1e9 if self.device.type == "cuda" else 0.0
            LOG.info("critic warmup %d/%d: rollouts done in %.1fs (mem=%.1f GB)", w, warmup_steps, t_roll, mem_after_roll)
            # Free engine KV cache before critic backward — same OOM concern as
            # the main step (engine + trainer share the CUDA caching allocator).
            if self.device.type == "cuda":
                self._torch.cuda.empty_cache()
            # Build MC returns from per-step rewards (gamma-discounted from terminal back)
            tokenized = []
            mc_returns_list = []
            for output in outputs:
                traj = tokenize_trajectory(
                    output.messages, tokenizer=self.tokenizer, max_length=cfg.max_total_length
                )
                if traj.n_assistant_turns == 0:
                    continue
                step_rewards, parse_flags = self._build_step_rewards(output)
                step_rewards = step_rewards[: traj.n_assistant_turns]
                parse_flags = parse_flags[: traj.n_assistant_turns]
                rewards_seq = build_per_token_rewards(
                    traj.assistant_spans, step_rewards, parse_flags, traj.seq_len, device=self.device,
                )
                # Monte-Carlo returns (no value bootstrapping)
                mc = self._torch.zeros_like(rewards_seq)
                running = 0.0
                for t in reversed(range(traj.seq_len)):
                    running = float(rewards_seq[t].item()) + cfg.algorithm.gamma * running
                    mc[t] = running
                tokenized.append(traj)
                mc_returns_list.append(mc)
            if not tokenized:
                continue
            batch = pad_trajectories(tokenized, pad_token_id=self.tokenizer.pad_token_id, device=self.device)
            seq_len = int(batch["input_ids"].shape[1])
            bsz = int(batch["input_ids"].shape[0])
            mc_returns = self._torch.zeros((bsz, seq_len), dtype=self._torch.float32, device=self.device)
            for i, mc in enumerate(mc_returns_list):
                mc_returns[i, : mc.shape[0]] = mc

            # Forward + value loss only
            self.optimizer.zero_grad(set_to_none=True)
            self.model.train()
            logits, values = self.model(batch["input_ids"], batch["attention_mask"], compute_values=True)
            values_aligned = values[:, :-1]
            targets = mc_returns[:, 1:]
            mask = batch["response_mask"][:, 1:]
            loss = 0.5 * masked_mean((values_aligned - targets).square(), mask)
            loss.backward()
            self.optimizer.step()
            t_total = time.perf_counter() - t_step
            mem_after_train = self._torch.cuda.memory_allocated() / 1e9 if self.device.type == "cuda" else 0.0
            value_loss_val = float(loss.item())
            # Log EVERY step (no longer every 10) so we can see live progress
            # in HF logs even when the wandb dashboard is the only real-time view.
            LOG.info(
                "critic warmup %d/%d: value_loss=%.4f roll=%.1fs total=%.1fs mem=%.1fGB n_traj=%d seq=%d",
                w, warmup_steps, value_loss_val, t_roll, t_total, mem_after_train, bsz, seq_len,
            )
            # Push warmup metrics to wandb so the dashboard isn't blank for the
            # entire warmup phase (which can be 50 steps × 3 min = 2.5h).
            # Use NEGATIVE step indices so warmup curves don't collide with PPO
            # step indices on the same wandb x-axis.
            if self._wandb is not None:
                self._wandb.log(
                    {
                        "warmup/value_loss": value_loss_val,
                        "warmup/rollout_secs": float(t_roll),
                        "warmup/total_secs": float(t_total),
                        "warmup/mem_gb": float(mem_after_train),
                        "warmup/n_traj": int(bsz),
                        "warmup/seq_len": int(seq_len),
                        "warmup/step": int(w),
                    },
                    step=-(warmup_steps - w + 1),  # negative = warmup, monotone-up
                )

        # Unfreeze actor
        for name, p in self.model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
        # Reset value head LR to the configured critic_lr.
        for group in self.optimizer.param_groups:
            if group.get("name") == "critic_value_head":
                group["lr"] = cfg.optimizer.critic_lr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _avg(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = set().union(*(r.keys() for r in records))
    return {k: float(sum(r.get(k, 0.0) for r in records)) / len(records) for k in keys}


__all__ = [
    "ActorCritic",
    "AlgorithmConfig",
    "CriticWarmupConfig",
    "EvalConfig",
    "LoRAConfig",
    "OptimizerConfig",
    "PPOTrainConfig",
    "PPOTrainer",
    "RolloutConfig",
    "StepMetrics",
]
