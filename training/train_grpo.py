"""Online GRPO/RLVR trainer for the DroneCaptureOps agent.

Why a custom trainer instead of TRL's GRPOTrainer?
  - Our episodes are multi-turn tool-calling sessions. Each rollout produces
    many (prompt, response) pairs that share one trajectory-level reward.
    TRL's single-shot interface does not match this shape.
  - The reward comes from a Python environment, not a HF reward model.
  - We want full control over advantage construction (group-relative within
    the same (task, seed) cell) so anti-gaming reward gates remain visible.

Pipeline per iteration:
  1. Build a sampling plan = list of (task_id, seed) cells × group_size.
  2. For every cell, run `group_size` rollouts using `LocalHFRLPolicy`.
     Each rollout:
       - resets the env at the cell's seed and task,
       - drives the policy through the standard `RolloutRunner`,
       - records per-turn (prompt_ids, response_ids) on the policy.
  3. Compute group-relative advantages from terminal rewards.
  4. For each (turn, advantage), compute log p_theta(response | prompt)
     under the current policy and the reference policy. Loss is
     `-(advantage * log_p) + kl_coef * (log_p - log_p_ref)^2 / 2`.
  5. Mini-batch gradient steps; save checkpoint every N iterations.

Pure-Python pieces (config, plan, advantage compute, summary stats) are
exposed at module scope and unit-tested in `tests/test_train_grpo.py`. The
heavy `train()` function imports torch/transformers/peft lazily so the rest
of this module is safe to import in CI without ML deps.

Usage:
    python -m training.train_grpo                        # uses defaults
    python -m training.train_grpo --config training/configs/rl_default.yaml
    python -m training.train_grpo --dry-run              # validate plan only
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "rl_default.yaml"

LOG = logging.getLogger("dronecaptureops.rl.train")


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------


class LoRAConfig(BaseModel):
    enabled: bool = True
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


class RLConfig(BaseModel):
    """Top-level RL trainer config."""

    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_name: str | None = None
    sft_adapter_path: str | None = None
    output_dir: str = "artifacts/rl-checkpoints"

    lora: LoRAConfig = Field(default_factory=LoRAConfig)

    train_tasks: list[str] = Field(default_factory=lambda: ["basic_thermal_survey"])
    eval_tasks: list[str] = Field(default_factory=list)

    tasks_per_iteration: int = Field(default=2, ge=1)
    seeds_per_task_iter: int = Field(default=1, ge=1)
    group_size: int = Field(default=4, ge=2)
    max_steps_per_episode: int = Field(default=30, ge=1)

    max_new_tokens: int = Field(default=384, ge=1)
    temperature: float = 0.9
    top_p: float = 0.95
    reward_field: str = "total"

    learning_rate: float = 1.0e-5
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_iterations: int = Field(default=10, ge=1)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    kl_coef: float = 0.05
    advantage_normalize: bool = True
    ppo_epochs: int = Field(default=1, ge=1)

    logging_steps: int = Field(default=1, ge=1)
    save_every_iterations: int = Field(default=5, ge=1)
    seed: int = 42


def load_config(path: Path | None) -> RLConfig:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        LOG.warning("config path %s not found, using built-in defaults", path)
        return RLConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    try:
        return RLConfig.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(f"invalid RL trainer config: {exc}") from exc


# ---------------------------------------------------------------------------
# Sampling plan
# ---------------------------------------------------------------------------


@dataclass
class RolloutCell:
    """One (task_id, seed) cell to roll out `group_size` times this iteration."""

    task_id: str
    seed: int
    group_size: int


def build_plan(config: RLConfig, *, iteration: int) -> list[RolloutCell]:
    """Pick `tasks_per_iteration` × `seeds_per_task_iter` cells for one iteration.

    Deterministic given (config.seed, iteration) so re-running an iteration
    after a crash sees the same data.
    """

    if not config.train_tasks:
        raise SystemExit("RL config has no train_tasks — add at least one task ID.")

    rng = random.Random((config.seed << 16) ^ iteration)
    cells: list[RolloutCell] = []
    for _ in range(config.tasks_per_iteration):
        task_id = rng.choice(config.train_tasks)
        for _ in range(config.seeds_per_task_iter):
            seed = rng.randrange(1, 1_000_000)
            cells.append(RolloutCell(task_id=task_id, seed=seed, group_size=config.group_size))
    return cells


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------


@dataclass
class CellRolloutResult:
    """Per-rollout outcome inside one cell, used to compute advantages."""

    cell: RolloutCell
    rollout_index: int
    reward: float
    success: bool
    steps: int
    parse_errors: int
    safety_gate: float
    integrity_gate: float
    turn_count: int


def compute_group_advantages(
    rewards: list[float], *, normalize: bool = True
) -> list[float]:
    """Group-relative advantages within a single (task, seed) cell.

    With normalize=True we use (r - mean) / (std + eps), the GRPO default.
    With normalize=False we just centre on the mean. Returns one value per
    rollout in the same order as the input.
    """

    if not rewards:
        return []
    mean = statistics.fmean(rewards)
    if not normalize or len(rewards) == 1:
        return [float(reward - mean) for reward in rewards]
    std = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
    if std < 1e-6:
        return [0.0 for _ in rewards]
    return [float((reward - mean) / std) for reward in rewards]


def aggregate_iteration_stats(results: list[CellRolloutResult]) -> dict[str, Any]:
    """Summary metrics for one RL iteration. Used in stdout/logs."""

    if not results:
        return {"n": 0}
    rewards = [r.reward for r in results]
    successes = [1.0 if r.success else 0.0 for r in results]
    parse_rate = sum(r.parse_errors for r in results) / max(sum(r.steps for r in results), 1)
    return {
        "n": len(results),
        "mean_reward": round(statistics.fmean(rewards), 4),
        "max_reward": round(max(rewards), 4),
        "min_reward": round(min(rewards), 4),
        "success_rate": round(statistics.fmean(successes), 4),
        "mean_steps": round(statistics.fmean(r.steps for r in results), 2),
        "parse_error_rate": round(parse_rate, 4),
        "mean_safety_gate": round(statistics.fmean(r.safety_gate for r in results), 4),
        "mean_integrity_gate": round(statistics.fmean(r.integrity_gate for r in results), 4),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: RLConfig) -> None:
    """Run the RL loop end-to-end.

    Heavy imports are local so the module-level config + plan + advantage
    helpers stay test-friendly without torch/transformers/peft installed.
    """

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "RL training requires `pip install -e '.[train]'` (torch, transformers, trl, peft)."
        ) from exc

    from dronecaptureops.agent.local_rl_policy import LocalHFRLPolicy
    from dronecaptureops.agent.rollout import RolloutResult, RolloutRunner
    from dronecaptureops.core.environment import DroneCaptureOpsEnvironment

    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("loading tokenizer: %s", config.tokenizer_name or config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name or config.model_name,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOG.info("loading model: %s", config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype="auto",
    )
    if config.sft_adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise SystemExit("sft_adapter_path requires `pip install peft`.") from exc
        LOG.info("loading SFT adapter: %s", config.sft_adapter_path)
        model = PeftModel.from_pretrained(model, config.sft_adapter_path)
    if config.lora.enabled:
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise SystemExit("lora.enabled requires `pip install peft`.") from exc
        if not hasattr(model, "peft_config"):
            peft_cfg = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                lora_dropout=config.lora.dropout,
                target_modules=list(config.lora.target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    reference_model: Any | None = None
    if config.kl_coef > 0:
        LOG.info("loading frozen reference model for KL penalty")
        reference_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype="auto",
        ).to(device)
        reference_model.eval()
        for param in reference_model.parameters():
            param.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(1, config.num_iterations + 1):
        LOG.info("=== iteration %d/%d ===", iteration, config.num_iterations)
        plan = build_plan(config, iteration=iteration)
        LOG.info("plan: %d cells × %d rollouts = %d episodes",
                 len(plan), config.group_size, len(plan) * config.group_size)

        cell_results, all_turns = _collect_rollouts(
            plan,
            config=config,
            model=model,
            tokenizer=tokenizer,
        )
        stats = aggregate_iteration_stats(cell_results)
        history.append({"iteration": iteration, "stats": stats})
        LOG.info("rollout stats: %s", json.dumps(stats, sort_keys=True))

        if not all_turns:
            LOG.warning("iteration %d collected no turns; skipping update", iteration)
            continue

        loss_value = _update_policy(
            turns=all_turns,
            model=model,
            tokenizer=tokenizer,
            reference_model=reference_model,
            optimizer=optimizer,
            config=config,
            device=device,
        )
        LOG.info("update loss=%.4f", loss_value)

        if iteration % config.save_every_iterations == 0:
            ckpt = output_dir / f"iter-{iteration}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt))
            tokenizer.save_pretrained(str(ckpt))
            LOG.info("saved checkpoint to %s", ckpt)

    final = output_dir / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    elapsed = time.time() - started
    LOG.info("training complete in %.1fs; final model at %s", elapsed, final)
    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2, sort_keys=True))


def _collect_rollouts(
    plan: list[RolloutCell],
    *,
    config: RLConfig,
    model: Any,
    tokenizer: Any,
) -> tuple[list[CellRolloutResult], list[dict[str, Any]]]:
    """Run every cell × group_size rollouts and tag each turn with an advantage.

    Returns a list of CellRolloutResult (for logging) and a flat list of
    annotated turn dicts ready for the gradient pass.
    """

    from dronecaptureops.agent.local_rl_policy import LocalHFRLPolicy
    from dronecaptureops.agent.rollout import RolloutRunner
    from dronecaptureops.core.environment import DroneCaptureOpsEnvironment

    cell_results: list[CellRolloutResult] = []
    all_turns: list[dict[str, Any]] = []

    for cell in plan:
        rewards: list[float] = []
        per_rollout_turns: list[list[dict[str, Any]]] = []
        per_rollout_meta: list[dict[str, Any]] = []
        for rollout_idx in range(cell.group_size):
            env = DroneCaptureOpsEnvironment()
            runner = RolloutRunner(env=env)
            policy = LocalHFRLPolicy(
                env=env,
                task_id=cell.task_id,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=True,
            )
            try:
                result = runner.run(
                    policy,
                    seed=cell.seed,
                    task_id=cell.task_id,
                    max_steps=config.max_steps_per_episode,
                )
            except Exception as exc:  # noqa: BLE001 — keep the iteration going
                LOG.exception("rollout failed for %s seed=%d: %s", cell.task_id, cell.seed, exc)
                continue
            reward = float(_extract_reward(result.reward_breakdown, config.reward_field))
            rewards.append(reward)
            policy_turns = list(policy.turns)
            per_rollout_turns.append([
                {
                    "prompt_token_ids": turn.prompt_token_ids,
                    "response_token_ids": turn.response_token_ids,
                    "parse_error": turn.parse_error,
                }
                for turn in policy_turns
                if turn.response_token_ids  # skip empty completions
            ])
            per_rollout_meta.append({
                "task_id": cell.task_id,
                "seed": cell.seed,
                "rollout_index": rollout_idx,
                "reward": reward,
                "success": bool(result.success),
                "steps": result.steps,
                "parse_errors": sum(1 for turn in policy_turns if turn.parse_error),
                "safety_gate": float(result.reward_breakdown.get("safety_gate", 1.0) or 0.0),
                "integrity_gate": float(result.reward_breakdown.get("integrity_gate", 1.0) or 0.0),
                "turn_count": len(policy_turns),
            })

        if not rewards:
            continue
        advantages = compute_group_advantages(
            rewards, normalize=config.advantage_normalize
        )
        for meta, turns_in_rollout, advantage in zip(per_rollout_meta, per_rollout_turns, advantages):
            cell_results.append(
                CellRolloutResult(
                    cell=cell,
                    rollout_index=meta["rollout_index"],
                    reward=meta["reward"],
                    success=meta["success"],
                    steps=meta["steps"],
                    parse_errors=meta["parse_errors"],
                    safety_gate=meta["safety_gate"],
                    integrity_gate=meta["integrity_gate"],
                    turn_count=meta["turn_count"],
                )
            )
            for turn in turns_in_rollout:
                all_turns.append({
                    **turn,
                    "advantage": float(advantage),
                    "task_id": meta["task_id"],
                    "seed": meta["seed"],
                })

    return cell_results, all_turns


def _extract_reward(breakdown: dict[str, Any], field: str) -> float:
    """Pull a reward scalar out of a RewardBreakdown dump.

    Defaults to terminal `total` if the requested field is missing or
    non-numeric, so a typo in config doesn't silently destroy training.
    """

    value = breakdown.get(field)
    if isinstance(value, int | float):
        return float(value)
    LOG.warning("reward_field %r missing or non-numeric in breakdown; falling back to `total`", field)
    fallback = breakdown.get("total", 0.0)
    return float(fallback if isinstance(fallback, int | float) else 0.0)


def _update_policy(
    *,
    turns: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    reference_model: Any | None,
    optimizer: Any,
    config: RLConfig,
    device: Any,
) -> float:
    """One pass of gradient descent over all collected turns."""

    import torch
    import torch.nn.functional as F

    model.train()
    total_loss = 0.0
    n_batches = 0
    pad_id = tokenizer.pad_token_id

    for ppo_epoch in range(config.ppo_epochs):
        order = list(range(len(turns)))
        random.shuffle(order)
        for batch_start in range(0, len(order), config.per_device_train_batch_size):
            batch_idx = order[batch_start : batch_start + config.per_device_train_batch_size]
            batch_loss = torch.tensor(0.0, device=device)
            for grad_step, idx in enumerate(batch_idx):
                turn = turns[idx]
                prompt_ids = turn["prompt_token_ids"]
                response_ids = turn["response_token_ids"]
                if not response_ids:
                    continue
                input_ids = torch.tensor(
                    [prompt_ids + response_ids], dtype=torch.long, device=device
                )
                attention_mask = torch.ones_like(input_ids)
                # Forward pass on full sequence; logits at position t predict token t+1.
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                # log p(response | prompt): align logits[prompt-1 : -1] with response_ids.
                prompt_len = len(prompt_ids)
                response_len = len(response_ids)
                response_logits = logits[0, prompt_len - 1 : prompt_len - 1 + response_len, :]
                response_targets = input_ids[0, prompt_len : prompt_len + response_len]
                log_probs = F.log_softmax(response_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    -1, response_targets.unsqueeze(-1)
                ).squeeze(-1)
                seq_log_prob = token_log_probs.sum()

                kl_term = torch.tensor(0.0, device=device)
                if reference_model is not None:
                    with torch.no_grad():
                        ref_logits = reference_model(
                            input_ids=input_ids, attention_mask=attention_mask
                        ).logits
                    ref_log_probs = F.log_softmax(
                        ref_logits[0, prompt_len - 1 : prompt_len - 1 + response_len, :],
                        dim=-1,
                    )
                    ref_token_log_probs = ref_log_probs.gather(
                        -1, response_targets.unsqueeze(-1)
                    ).squeeze(-1)
                    ref_seq_log_prob = ref_token_log_probs.sum()
                    kl_term = (seq_log_prob - ref_seq_log_prob) ** 2 / 2.0

                advantage = float(turn["advantage"])
                loss = -(advantage * seq_log_prob) + config.kl_coef * kl_term
                batch_loss = batch_loss + loss / max(len(batch_idx), 1)

            if batch_idx:
                batch_loss = batch_loss / max(config.gradient_accumulation_steps, 1)
                batch_loss.backward()
                if (n_batches + 1) % config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        config.max_grad_norm,
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                total_loss += float(batch_loss.detach().cpu())
                n_batches += 1
    if n_batches:
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config.max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Online GRPO/RLVR trainer for DroneCaptureOps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model", dest="model_name", default=None)
    parser.add_argument("--sft-adapter", default=None, help="Path to an SFT LoRA adapter to warm-start from.")
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--no-kl", action="store_true", help="Disable KL penalty (skip reference forward).")
    parser.add_argument("--dry-run", action="store_true", help="Validate config + plan only.")
    return parser.parse_args()


def _apply_cli_overrides(config: RLConfig, args: argparse.Namespace) -> RLConfig:
    update: dict[str, Any] = {}
    for attr in ("output_dir", "model_name", "num_iterations", "group_size", "seed"):
        value = getattr(args, attr, None)
        if value is not None:
            update[attr] = value
    if getattr(args, "sft_adapter", None) is not None:
        update["sft_adapter_path"] = args.sft_adapter
    if args.no_lora:
        update["lora"] = config.lora.model_copy(update={"enabled": False})
    if args.no_kl:
        update["kl_coef"] = 0.0
    if not update:
        return config
    return config.model_copy(update=update)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)

    LOG.info(
        "config: model=%s lora=%s iters=%d tasks_per_iter=%d group=%d max_steps=%d kl=%g",
        config.model_name,
        config.lora.enabled,
        config.num_iterations,
        config.tasks_per_iteration,
        config.group_size,
        config.max_steps_per_episode,
        config.kl_coef,
    )

    if args.dry_run:
        plans = [build_plan(config, iteration=i) for i in range(1, config.num_iterations + 1)]
        flat = [
            {"iteration": i + 1, "task_id": cell.task_id, "seed": cell.seed, "group_size": cell.group_size}
            for i, cells in enumerate(plans)
            for cell in cells
        ]
        print(json.dumps({
            "model": config.model_name,
            "iterations": config.num_iterations,
            "cells_per_iteration": config.tasks_per_iteration * config.seeds_per_task_iter,
            "group_size": config.group_size,
            "total_episodes": len(flat) * config.group_size,
            "kl_coef": config.kl_coef,
            "lora_enabled": config.lora.enabled,
            "first_5_cells": flat[:5],
        }, indent=2, sort_keys=True))
        return 0

    train(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
