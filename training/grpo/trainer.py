"""Single-process GRPO trainer for DroneCaptureOps (no vLLM).

Architecture:
    - 1 GPU, 1 process, no Ray.
    - Policy: Qwen base + LoRA adapter (continued from SFT warmstart).
    - Reference: SAME backbone with adapter disabled via PEFT
      ``disable_adapter()`` — saves us a second model copy.
    - Rollouts: ``model.generate()`` directly, no vLLM, no engine
      bootstrap, no Fabric Manager probe needed beyond standard CUDA
      availability.
    - GRPO step:
        1. sample N prompts, G rollouts each (sequential).
        2. tokenize each rollout.
        3. forward pass for old log-probs (no grad).
        4. forward pass with adapter disabled for ref log-probs.
        5. group-normalize the per-rollout rewards into advantages.
        6. K-epoch update: forward new log-probs, GRPO loss + KL +
           entropy, backward, clip-grad, step.

The math (``compute_approx_kl``, ``group_advantages``,
``grpo_policy_loss``) lives in ``loss.py``.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS
from training.grpo.config import GRPOTrainConfig
from training.grpo.loss import (
    compute_approx_kl,
    entropy_from_logits,
    gather_token_log_probs,
    group_advantages,
    grpo_policy_loss,
    masked_mean,
)
from training.grpo.reporting import (
    append_jsonl,
    build_training_report,
    rollout_record,
    write_step_report,
)
from training.grpo.rollout_pool import (
    GRPORolloutOutput,
    GRPORolloutSpec,
    run_rollout_batch,
    sample_specs,
)
from training.ppo.tokenization import (
    TokenizedTrajectory,
    pad_trajectories,
    tokenize_trajectory,
)


LOG = logging.getLogger("dronecaptureops.grpo.trainer")


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
    advantage_mean: float = 0.0
    advantage_std: float = 0.0
    policy_loss: float = 0.0
    clip_frac: float = 0.0
    approx_kl: float = 0.0
    kl_to_ref: float = 0.0
    entropy: float = 0.0
    ratio_mean: float = 1.0
    grad_norm: float = 0.0


class GRPOTrainer:
    def __init__(self, config: GRPOTrainConfig):
        self.cfg = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.rollouts_path = self.output_dir / config.reporting.rollouts_jsonl
        self.traces_dir = self.output_dir / config.reporting.traces_dir

        # Lazy heavyweight imports — keeps tests fast on CPU-only hosts.
        import torch as _torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = _torch
        self._PeftModel = PeftModel
        self._get_peft_model = get_peft_model
        self._LoraConfig = LoraConfig

        try:
            _torch.cuda.init()
            device_count = _torch.cuda.device_count()
        except Exception as exc:
            raise RuntimeError(
                "GRPO training requires a fully initialized CUDA runtime. "
                "CUDA was visible but failed runtime initialization."
            ) from exc
        if device_count < 1:
            raise RuntimeError(
                "GRPO training requires CUDA. Refusing to continue on CPU because "
                "that would silently load the training model in float32 and waste "
                "remote job time."
            )
        self.device = _torch.device("cuda")
        self.dtype = _torch.bfloat16 if self.device.type == "cuda" else _torch.float32

        LOG.info("loading tokenizer: %s", config.tokenizer_name or config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name or config.model_name,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Group-batched rollouts call ``model.generate()`` on G prompts at
        # once. HF generate requires LEFT padding for decoder-only LMs so
        # all prompts end at the same column when generation begins.
        self.tokenizer.padding_side = "left"

        LOG.info(
            "loading base model: %s (%s, attn=sdpa)",
            config.model_name,
            self.dtype,
        )
        base = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=self.dtype,
            attn_implementation="sdpa",
        )
        base.gradient_checkpointing_enable()

        # Continue from SFT LoRA, or attach a fresh adapter only when the
        # config explicitly opts into infrastructure-only fresh-LoRA smoke.
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
                    load_target, load_kwargs = sft_spec, {}
            try:
                load_kwargs = {**load_kwargs, "torch_device": "cpu"}
                LOG.info("loading SFT LoRA: %s (kwargs=%s)", load_target, load_kwargs)
                policy = PeftModel.from_pretrained(base, load_target, is_trainable=True, **load_kwargs)
                sft_loaded = True
                LOG.info("loaded SFT LoRA warm-start from %s", sft_spec)
            except Exception as exc:
                if not config.allow_fresh_lora:
                    raise RuntimeError(
                        f"SFT checkpoint {sft_spec!r} could not be loaded. "
                        "Refusing to start GRPO from fresh LoRA; set "
                        "allow_fresh_lora: true only for infrastructure smoke tests."
                    ) from exc
                LOG.warning(
                    "SFT checkpoint %s could not be loaded (%s); using explicit fresh LoRA.",
                    sft_spec,
                    exc,
                )
        if not sft_loaded:
            if not config.allow_fresh_lora:
                raise RuntimeError(
                    "No SFT checkpoint was loaded. Set sft_checkpoint to a valid "
                    "local path or Hub repo_id, or set allow_fresh_lora: true "
                    "only for infrastructure smoke tests."
                )
            LOG.warning("using explicit fresh LoRA: r=%d alpha=%d", config.lora.rank, config.lora.alpha)
            lora_cfg = LoraConfig(
                r=config.lora.rank,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=list(config.lora.target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            policy = get_peft_model(base, lora_cfg)

        self.model = policy.to(self.device)

        # LoRA params are the only trainable surface (no value head in GRPO).
        for name, p in self.model.named_parameters():
            p.requires_grad = "lora_" in name

        from torch.optim import AdamW

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable,
            lr=config.optimizer.actor_lr,
            betas=tuple(config.optimizer.betas),
            weight_decay=config.optimizer.weight_decay,
        )

        # Wandb (optional; same pattern as PPO trainer).
        self._wandb = None
        if config.wandb_mode != "disabled":
            try:
                import wandb

                self._wandb = wandb
                self._wandb.init(
                    project=config.wandb_project or "dronecaptureops-grpo",
                    name=config.wandb_run_name,
                    mode=config.wandb_mode,
                    config=config.model_dump(),
                )
            except ImportError:
                LOG.warning("wandb not installed; logging to JSONL only")
                self._wandb = None

    # ---------- Task sampling ----------

    def _resolved_train_tasks(self) -> list[str]:
        if self.cfg.train_tasks:
            return list(self.cfg.train_tasks)
        held_out = set(self.cfg.eval.held_out_tasks)
        all_ids = [task_id for task_id in SOLAR_TASKS if task_id not in held_out]
        if not all_ids:
            raise RuntimeError("no training tasks resolved — every task is held out")
        return all_ids

    # ---------- Forward pass helpers ----------

    def _forward_logprobs(
        self,
        batch: dict[str, torch.Tensor],
        *,
        train_mode: bool = False,
        compute_entropy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        was_training = self.model.training
        if train_mode:
            self.model.train()
        else:
            self.model.eval()
        try:
            ctx = torch.enable_grad() if train_mode else torch.no_grad()
            with ctx:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                logits = outputs.logits
                logprobs = gather_token_log_probs(logits, batch["input_ids"])
                entropy = entropy_from_logits(logits[:, :-1, :]) if compute_entropy else None
        finally:
            self.model.train(was_training)
        return logprobs, entropy

    def _forward_ref_logprobs(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                with self.model.disable_adapter():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False,
                    )
                    return gather_token_log_probs(outputs.logits, batch["input_ids"])
        finally:
            self.model.train(was_training)

    def _forward_logprobs_chunked(
        self,
        batch: dict[str, torch.Tensor],
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        """No-grad logprobs computed in batch-dim chunks.

        ``outputs.logits`` is ``[B, T, V]``. For Qwen3-4B with ``V=152K``,
        a single forward over the full B=32 batch padded to a few thousand
        tokens trips OOM even on H200. Chunking the batch and freeing the
        intermediate logits between chunks keeps peak memory bounded.
        """
        torch_ = self._torch
        bsz = int(batch["input_ids"].shape[0])
        chunk_size = max(1, int(chunk_size))
        chunks: list[torch.Tensor] = []
        for start in range(0, bsz, chunk_size):
            end = min(start + chunk_size, bsz)
            sub = {k: v[start:end] for k, v in batch.items()}
            lp, _ = self._forward_logprobs(sub, train_mode=False, compute_entropy=False)
            chunks.append(lp.detach())
            del lp, sub
            if torch_.cuda.is_available():
                torch_.cuda.empty_cache()
        return torch_.cat(chunks, dim=0)

    def _forward_ref_logprobs_chunked(
        self,
        batch: dict[str, torch.Tensor],
        *,
        chunk_size: int,
    ) -> torch.Tensor:
        torch_ = self._torch
        bsz = int(batch["input_ids"].shape[0])
        chunk_size = max(1, int(chunk_size))
        chunks: list[torch.Tensor] = []
        for start in range(0, bsz, chunk_size):
            end = min(start + chunk_size, bsz)
            sub = {k: v[start:end] for k, v in batch.items()}
            lp = self._forward_ref_logprobs(sub)
            chunks.append(lp.detach())
            del lp, sub
            if torch_.cuda.is_available():
                torch_.cuda.empty_cache()
        return torch_.cat(chunks, dim=0)

    # ---------- Main training step ----------

    def step(self, step_idx: int, rng: random.Random) -> StepMetrics:
        cfg = self.cfg
        torch_ = self._torch

        # 1. Sample specs and run rollouts sequentially.
        train_tasks = self._resolved_train_tasks()
        specs = sample_specs(
            rng=rng,
            train_tasks=train_tasks,
            prompts_per_step=cfg.rollout.prompts_per_step,
            group_size=cfg.rollout.group_size,
        )
        t0 = time.perf_counter()
        outputs_with_none = run_rollout_batch(
            specs,
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=cfg.rollout.temperature,
            top_p=cfg.rollout.top_p,
            max_tokens=cfg.rollout.max_new_tokens_per_turn,
            max_history_steps=cfg.rollout.max_history_steps,
            max_steps=cfg.rollout.max_episode_steps,
        )
        t_rollout = time.perf_counter() - t0
        if torch_.cuda.is_available():
            torch_.cuda.empty_cache()
        # Keep ordering aligned with specs but drop failures.
        outputs: list[GRPORolloutOutput] = [o for o in outputs_with_none if o is not None]
        if not outputs:
            LOG.warning("step %d produced 0 rollouts; skipping update", step_idx)
            metrics = self._summarize_metrics(step_idx, outputs, outputs, t_rollout, 0.0, 0.0)
            self._write_rollout_diagnostics(step_idx, metrics, outputs)
            return metrics

        # 2. Tokenize trajectories. Drop empties.
        tokenized: list[TokenizedTrajectory] = []
        kept_outputs: list[GRPORolloutOutput] = []
        for output in outputs:
            traj = tokenize_trajectory(
                output.messages,
                tokenizer=self.tokenizer,
                max_length=cfg.max_total_length,
            )
            if traj.n_assistant_turns == 0:
                continue
            tokenized.append(traj)
            kept_outputs.append(output)

        if not tokenized:
            LOG.warning("step %d produced 0 trainable trajectories; skipping update", step_idx)
            metrics = self._summarize_metrics(step_idx, outputs, kept_outputs, t_rollout, 0.0, 0.0)
            self._write_rollout_diagnostics(step_idx, metrics, outputs)
            return metrics

        # 3. Compute group-normalized advantages from rollout total_reward.
        rewards_per_kept = torch_.tensor(
            [float(o.result.total_reward) for o in kept_outputs],
            dtype=torch_.float32,
            device=self.device,
        )
        # Group structure may have been broken by drops; rebuild groups by
        # prompt_index and pad rewards to the original group size with the
        # group's own mean so the group normalization still produces zero
        # advantage for the missing slot. This is a pragmatic safety net so
        # one bad rollout cannot destroy the whole step.
        advantages_per_kept = self._group_normalize_rewards(rewards_per_kept, kept_outputs)

        # 4. Build padded batch + response masks.
        batch = pad_trajectories(
            tokenized,
            pad_token_id=self.tokenizer.pad_token_id,
            device=self.device,
        )
        bsz = int(batch["input_ids"].shape[0])

        # 5. Forward pass for old log-probs and ref log-probs (no grad).
        # Chunk along batch dim to keep peak memory bounded — a single
        # forward over the full padded batch materializes
        # ``[B, T, vocab]`` logits which OOMs even on H200 for V≈150K.
        t0 = time.perf_counter()
        chunk_size = max(1, int(cfg.micro_batch_size))
        old_logprobs = self._forward_logprobs_chunked(batch, chunk_size=chunk_size)
        if torch_.cuda.is_available():
            torch_.cuda.empty_cache()
        ref_logprobs = self._forward_ref_logprobs_chunked(batch, chunk_size=chunk_size)
        if torch_.cuda.is_available():
            torch_.cuda.empty_cache()
        t_forward = time.perf_counter() - t0

        response_mask_shifted = batch["response_mask"][:, 1:]

        # 6. K-epoch GRPO update.
        t0 = time.perf_counter()
        all_indices = list(range(bsz))
        update_metrics: list[dict[str, float]] = []
        for _epoch in range(cfg.grpo_epochs):
            rng.shuffle(all_indices)
            for start in range(0, bsz, cfg.minibatch_size):
                idx = all_indices[start : start + cfg.minibatch_size]
                if not idx:
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                accum_metrics: list[dict[str, float]] = []
                for micro_start in range(0, len(idx), cfg.micro_batch_size):
                    micro = idx[micro_start : micro_start + cfg.micro_batch_size]
                    micro_batch = {k: v[micro] for k, v in batch.items()}
                    micro_old_logprobs = old_logprobs[micro]
                    micro_ref_logprobs = ref_logprobs[micro]
                    micro_response_mask = micro_batch["response_mask"][:, 1:]
                    micro_advantages = advantages_per_kept[micro]

                    new_logprobs, new_entropy = self._forward_logprobs(
                        micro_batch, train_mode=True, compute_entropy=True
                    )
                    policy_loss, p_metrics = grpo_policy_loss(
                        new_logprobs,
                        micro_old_logprobs,
                        micro_advantages,
                        response_mask=micro_response_mask,
                        eps_low=cfg.algorithm.eps_clip_low,
                        eps_high=cfg.algorithm.eps_clip_high,
                    )
                    kl = compute_approx_kl(
                        new_logprobs,
                        micro_ref_logprobs,
                        estimator=cfg.algorithm.kl_estimator,
                        mask=micro_response_mask,
                    )
                    kl_term = masked_mean(kl, micro_response_mask)
                    entropy_term = (
                        masked_mean(new_entropy, micro_response_mask)
                        if new_entropy is not None
                        else torch_.zeros((), device=self.device)
                    )
                    loss = (
                        policy_loss
                        + cfg.algorithm.kl_coef * kl_term
                        - cfg.algorithm.entropy_coef * entropy_term
                    )
                    accum_scale = float(len(micro)) / float(len(idx))
                    (loss * accum_scale).backward()
                    accum_metrics.append({
                        **p_metrics,
                        "kl_to_ref": float(kl_term.item()),
                        "entropy": float(entropy_term.item()),
                    })

                grad_norm = torch_.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    cfg.optimizer.max_grad_norm,
                )
                self.optimizer.step()
                update_metrics.append({**_avg(accum_metrics), "grad_norm": float(grad_norm)})
                if torch_.cuda.is_available():
                    torch_.cuda.empty_cache()
        t_update = time.perf_counter() - t0

        metrics = self._summarize_metrics(step_idx, outputs, kept_outputs, t_rollout, t_forward, t_update)
        # Stitch advantage stats and update metrics into the row.
        with torch_.no_grad():
            metrics.advantage_mean = float(advantages_per_kept.mean().item())
            metrics.advantage_std = float(advantages_per_kept.std(unbiased=False).item())
        if update_metrics:
            avg = _avg(update_metrics)
            metrics.policy_loss = avg.get("policy_loss", 0.0)
            metrics.clip_frac = avg.get("clip_frac", 0.0)
            metrics.approx_kl = avg.get("approx_kl", 0.0)
            metrics.kl_to_ref = avg.get("kl_to_ref", 0.0)
            metrics.entropy = avg.get("entropy", 0.0)
            metrics.ratio_mean = avg.get("ratio_mean", 1.0)
            metrics.grad_norm = avg.get("grad_norm", 0.0)
        self._write_rollout_diagnostics(step_idx, metrics, outputs)
        return metrics

    # ---------- Aggregation ----------

    def _group_normalize_rewards(
        self,
        rewards: torch.Tensor,
        kept_outputs: list[GRPORolloutOutput],
    ) -> torch.Tensor:
        """Group-normalize rewards using ``prompt_index`` as the group key.

        Robust to dropped rollouts: groups smaller than 2 produce
        zero advantage (no signal but no NaN). Groups with constant
        rewards also produce zero (because std == 0).
        """

        torch_ = self._torch
        adv = torch_.zeros_like(rewards)
        groups: dict[int, list[int]] = {}
        for i, output in enumerate(kept_outputs):
            groups.setdefault(int(output.spec.prompt_index), []).append(i)
        eps = self.cfg.algorithm.adv_eps
        for indices in groups.values():
            if len(indices) < 2:
                continue
            sub = rewards[indices]
            mean = sub.mean()
            std = sub.std(unbiased=False)
            if float(std.item()) < 1e-8:
                continue
            adv[indices] = (sub - mean) / (std + eps)
        if self.cfg.algorithm.clip_advantage:
            clip = self.cfg.algorithm.advantage_clip
            adv = adv.clamp(-clip, clip)
        return adv

    def _summarize_metrics(
        self,
        step_idx: int,
        all_outputs: list[GRPORolloutOutput],
        kept_outputs: list[GRPORolloutOutput],
        t_rollout: float,
        t_forward: float,
        t_update: float,
    ) -> StepMetrics:
        n = len(all_outputs)
        if n == 0:
            return StepMetrics(
                step=step_idx,
                rollout_secs=t_rollout,
                forward_secs=t_forward,
                update_secs=t_update,
                n_rollouts=0,
                n_assistant_turns=0,
                mean_episode_steps=0.0,
                mean_total_reward=0.0,
                mean_terminal_reward=0.0,
                success_rate=0.0,
                parse_error_rate=0.0,
            )
        n_turns = sum(len(o.result.trajectory) for o in kept_outputs)
        success = sum(1 for o in all_outputs if o.result.success) / n
        mean_steps = sum(o.result.steps for o in all_outputs) / n
        mean_reward = sum(o.result.total_reward for o in all_outputs) / n
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
            mean_terminal_reward=float(mean_reward),
            success_rate=float(success),
            parse_error_rate=float(parse_errors / total_steps),
        )

    # ---------- Persistence ----------

    def save_checkpoint(self, step: int, *, label: str = "step") -> Path:
        ckpt_dir = self.output_dir / f"{label}_{step}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(ckpt_dir / "adapter"))
        self.tokenizer.save_pretrained(str(ckpt_dir / "tokenizer"))
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
            "step %d | reward=%.3f success=%.2f parse_err=%.2f kl=%.3f ent=%.3f ploss=%.3f adv_mean=%.3f t_roll=%.1fs t_upd=%.1fs",
            metrics.step,
            metrics.mean_total_reward,
            metrics.success_rate,
            metrics.parse_error_rate,
            metrics.kl_to_ref,
            metrics.entropy,
            metrics.policy_loss,
            metrics.advantage_mean,
            metrics.rollout_secs,
            metrics.update_secs,
        )

    def _write_rollout_diagnostics(
        self,
        step_idx: int,
        metrics: StepMetrics,
        outputs: list[GRPORolloutOutput],
    ) -> None:
        if not self.cfg.reporting.enabled:
            return
        try:
            rows = []
            for i, output in enumerate(outputs):
                row = rollout_record(
                    output,
                    step=step_idx,
                    rollout_index=i,
                    include_messages=self.cfg.reporting.include_messages,
                )
                # Stitch GRPO-specific group bookkeeping for downstream reports.
                row["prompt_index"] = int(output.spec.prompt_index)
                row["group_index"] = int(output.spec.group_index)
                rows.append(row)
            append_jsonl(self.rollouts_path, rows)
            write_step_report(
                self.output_dir,
                step=step_idx,
                metrics=metrics.__dict__,
                rollouts=rows,
            )
            self._write_trace_samples(step_idx, outputs)
        except Exception:  # noqa: BLE001
            LOG.exception("failed to write GRPO rollout diagnostics for step %d", step_idx)

    def _write_trace_samples(self, step_idx: int, outputs: list[GRPORolloutOutput]) -> None:
        n = max(0, self.cfg.reporting.max_trace_samples_per_step)
        if n == 0 or not outputs:
            return
        from dronecaptureops.evaluation.tracing import write_trace_artifacts

        for i, output in enumerate(outputs[:n]):
            trace_dir = self.traces_dir / f"step_{step_idx:04d}" / f"rollout_{i:02d}"
            payload = output.result.model_dump(mode="json")
            write_trace_artifacts(payload, trace_dir)

    # ---------- Top-level fit() ----------

    def fit(self) -> None:
        cfg = self.cfg
        rng = random.Random(cfg.seed)

        for step in range(1, cfg.total_steps + 1):
            metrics = self.step(step, rng)
            self.log_metrics(metrics)
            if step % cfg.save_interval_steps == 0 or step == cfg.total_steps:
                self.save_checkpoint(step)

        self.save_checkpoint(cfg.total_steps, label="final")
        if cfg.reporting.enabled:
            try:
                build_training_report(self.output_dir)
                LOG.info("training report written under %s", self.output_dir / "reports")
            except Exception:  # noqa: BLE001
                LOG.exception("failed to build GRPO training report")
        if self._wandb is not None:
            self._wandb.finish()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _avg(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    keys = set().union(*(r.keys() for r in records))
    return {k: float(sum(r.get(k, 0.0) for r in records)) / len(records) for k in keys}


__all__ = [
    "GRPOTrainConfig",
    "GRPOTrainer",
    "StepMetrics",
]
