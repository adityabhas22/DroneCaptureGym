"""Pydantic config schema for the GRPO trainer.

Kept separate from ``trainer.py`` so ``--dry-run``, ``--help``, and CI
config validation don't pay the torch import cost.

GRPO (Group Relative Policy Optimization, DeepSeek-R1) drops the value
head and computes advantages by normalizing rewards within G samples
per prompt. That removes:

- the value head and its forward/backward pass;
- the critic warmup phase (which was the L40S OOM site for PPO);
- token-level GAE bookkeeping.

Everything related to vLLM is also gone — this branch generates with
HF transformers ``model.generate()`` only.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LoRAConfig(BaseModel):
    rank: int = 64
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])


class OptimizerConfig(BaseModel):
    actor_lr: float = 5.0e-6
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_steps: int = 0
    scheduler: str = "cosine"


class AlgorithmConfig(BaseModel):
    """GRPO update knobs.

    No GAE — advantages come from group-normalized rewards. ``kl_coef``
    is the loss-side coefficient on KL(π || π_ref) computed via the k3
    estimator (same as PPO trainer). ``adv_eps`` is the standardization
    epsilon for group normalization.
    """

    eps_clip_low: float = 0.2
    eps_clip_high: float = 0.2
    kl_coef: float = 0.01
    kl_estimator: str = "k3"
    entropy_coef: float = 0.001
    adv_eps: float = 1.0e-4
    clip_advantage: bool = True
    advantage_clip: float = 5.0


class RolloutConfig(BaseModel):
    """Sampling shape for one GRPO step.

    A single step runs ``prompts_per_step`` distinct prompts (= task
    scenarios). For each prompt we sample ``group_size`` rollouts so we
    can group-normalize their rewards into advantages. Total rollouts
    per step = prompts_per_step * group_size.
    """

    prompts_per_step: int = 4
    group_size: int = 4
    temperature: float = 0.9
    top_p: float = 1.0
    max_new_tokens_per_turn: int = 384
    max_episode_steps: int = 8
    max_history_steps: int = 4


class EvalConfig(BaseModel):
    enabled: bool = False
    interval_steps: int = 25
    held_out_tasks: list[str] = Field(default_factory=list)
    seeds_per_task: int = 1
    max_episode_steps: int = 8


class ReportingConfig(BaseModel):
    enabled: bool = True
    rollouts_jsonl: str = "rollouts/rollouts.jsonl"
    reports_dir: str = "reports"
    traces_dir: str = "traces"
    max_trace_samples_per_step: int = 2
    include_messages: bool = False


class GRPOTrainConfig(BaseModel):
    """Top-level GRPO config — every knob in one place."""

    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer_name: str | None = None
    sft_checkpoint: str | None = None
    allow_fresh_lora: bool = False
    output_dir: str = "artifacts/grpo-checkpoints"

    max_prompt_length: int = 2048
    max_total_length: int = 4096
    pad_to_multiple_of: int = 8

    grpo_epochs: int = 1
    minibatch_size: int = 1
    micro_batch_size: int = 1
    total_steps: int = 10
    save_interval_steps: int = 5
    log_interval_steps: int = 1

    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    train_tasks: list[str] | None = None
    seed: int = 42

    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "disabled"


__all__ = [
    "AlgorithmConfig",
    "EvalConfig",
    "GRPOTrainConfig",
    "LoRAConfig",
    "OptimizerConfig",
    "ReportingConfig",
    "RolloutConfig",
]
