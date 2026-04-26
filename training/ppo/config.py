"""Pydantic config schema for the PPO trainer.

Kept separate from `trainer.py` so `--dry-run`, `--help`, and CI-side
config validation don't pay the torch import cost.
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
    actor_lr: float = 1.0e-5
    critic_lr: float = 1.0e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0
    warmup_steps: int = 10
    scheduler: str = "cosine"


class AlgorithmConfig(BaseModel):
    gamma: float = 0.99
    lam: float = 0.95
    eps_clip_low: float = 0.2
    eps_clip_high: float = 0.2
    value_clip: float = 0.2
    dual_clip_c: float | None = None
    kl_coef: float = 0.01
    kl_estimator: str = "k3"
    entropy_coef: float = 0.001
    whiten_advantages: bool = True


class RolloutConfig(BaseModel):
    rollout_batch_size: int = 16
    max_workers: int = 16
    temperature: float = 0.7
    top_p: float = 1.0
    max_new_tokens_per_turn: int = 1024
    max_episode_steps: int = 40
    max_history_steps: int = 24


class CriticWarmupConfig(BaseModel):
    enabled: bool = True
    steps: int = 50
    lr_multiplier: float = 5.0


class EvalConfig(BaseModel):
    enabled: bool = True
    interval_steps: int = 25
    held_out_tasks: list[str] = Field(default_factory=list)
    seeds_per_task: int = 3
    max_episode_steps: int = 40


class ReportingConfig(BaseModel):
    enabled: bool = True
    rollouts_jsonl: str = "rollouts/rollouts.jsonl"
    reports_dir: str = "reports"
    traces_dir: str = "traces"
    max_trace_samples_per_step: int = 2
    include_messages: bool = False


class PPOTrainConfig(BaseModel):
    """Top-level config — every knob in one place."""

    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer_name: str | None = None
    sft_checkpoint: str | None = None
    allow_fresh_lora: bool = False
    output_dir: str = "artifacts/ppo-checkpoints"

    max_prompt_length: int = 8192
    max_total_length: int = 32768
    pad_to_multiple_of: int = 8

    ppo_epochs: int = 2
    minibatch_size: int = 4
    micro_batch_size: int = 1
    total_steps: int = 100
    save_interval_steps: int = 25
    log_interval_steps: int = 1

    vllm_gpu_memory_utilization: float = 0.4
    vllm_max_model_len: int = 32768
    vllm_dtype: str = "bfloat16"
    vllm_enforce_eager: bool = False
    vllm_lora_dir: str = "artifacts/ppo-lora-cache"

    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    critic_warmup: CriticWarmupConfig = Field(default_factory=CriticWarmupConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    train_tasks: list[str] | None = None
    seed: int = 42

    wandb_project: str | None = None
    wandb_run_name: str | None = None
    wandb_mode: str = "disabled"


__all__ = [
    "AlgorithmConfig",
    "CriticWarmupConfig",
    "EvalConfig",
    "LoRAConfig",
    "OptimizerConfig",
    "PPOTrainConfig",
    "ReportingConfig",
    "RolloutConfig",
]
