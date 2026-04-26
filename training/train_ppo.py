"""PPO trainer entrypoint — CLI matching `training/sft_warmstart.py`.

Mirrors the SFT entrypoint's CLI shape so HF Jobs and the local dev
loop can drive both the same way:

    python -m training.train_ppo                                     # built-in defaults
    python -m training.train_ppo --config training/configs/ppo_train_default.yaml
    python -m training.train_ppo --model Qwen/Qwen2.5-3B-Instruct --output-dir artifacts/ppo

The HF Jobs entrypoint at `training/hf_jobs/entrypoint.py:148-160`
invokes us as `python -m training.train_ppo --config <yaml> --model
<base> --output-dir <out>`, so this CLI is the contract surface.

Heavy imports (torch / transformers / peft / vllm) are deferred into
`PPOTrainer.__init__` so `--dry-run` and `--help` stay fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from training.env_utils import hf_token_visible, load_dotenv_if_present, visible_token_names
from training.ppo.config import PPOTrainConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "ppo_train_default.yaml"

LOG = logging.getLogger("dronecaptureops.ppo.train")


def configure_gpu_runtime_env() -> None:
    """Set CUDA/vLLM safety env vars before importing torch/vLLM.

    PPO initializes PyTorch in the parent process and then starts vLLM for
    rollouts. vLLM must therefore use spawn workers; fork can crash with
    "Cannot re-initialize CUDA in forked subprocess", especially on H200.
    """

    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def load_config(path: Path | None) -> PPOTrainConfig:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        LOG.warning("config path %s not found, using built-in defaults", path)
        return PPOTrainConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    try:
        return PPOTrainConfig.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(f"invalid PPO trainer config: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PPO trainer for DroneCaptureOps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", dest="model_name", default=None, help="Override base model.")
    parser.add_argument("--sft-checkpoint", default=None, help="Override SFT LoRA path.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total PPO steps.")
    parser.add_argument("--rollout-batch-size", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--critic-lr", type=float, default=None)
    parser.add_argument("--kl-coef", type=float, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-critic-warmup",
        action="store_true",
        help="Skip the value-head warmup phase (useful for smoke tests).",
    )
    parser.add_argument(
        "--no-vllm",
        action="store_true",
        help="Validate config + model loading without starting vLLM. Used for unit/CI smoke.",
    )
    parser.add_argument(
        "--allow-fresh-lora",
        action="store_true",
        help="Allow PPO to start from a fresh LoRA when no SFT checkpoint is loaded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and resolve task list without launching training.",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default=None,
        help="Override wandb logging mode.",
    )
    parser.add_argument(
        "--wandb-run-name",
        default=None,
        help="Override wandb run name.",
    )
    return parser.parse_args()


def _apply_cli_overrides(config: PPOTrainConfig, args: argparse.Namespace) -> PPOTrainConfig:
    update: dict[str, Any] = {}
    if args.model_name:
        update["model_name"] = args.model_name
    if args.sft_checkpoint:
        update["sft_checkpoint"] = args.sft_checkpoint
    if args.output_dir:
        update["output_dir"] = args.output_dir
    if args.total_steps is not None:
        update["total_steps"] = args.total_steps
    if args.seed is not None:
        update["seed"] = args.seed
    if args.wandb_mode is not None:
        update["wandb_mode"] = args.wandb_mode
    if args.wandb_run_name is not None:
        update["wandb_run_name"] = args.wandb_run_name

    # Composed overrides — copy the sub-config then update.
    if args.actor_lr is not None or args.critic_lr is not None:
        opt = config.optimizer.model_copy(update={
            **({"actor_lr": args.actor_lr} if args.actor_lr is not None else {}),
            **({"critic_lr": args.critic_lr} if args.critic_lr is not None else {}),
        })
        update["optimizer"] = opt
    if args.kl_coef is not None:
        update["algorithm"] = config.algorithm.model_copy(update={"kl_coef": args.kl_coef})
    if args.rollout_batch_size is not None:
        update["rollout"] = config.rollout.model_copy(update={"rollout_batch_size": args.rollout_batch_size})
    if args.minibatch_size is not None:
        update["minibatch_size"] = args.minibatch_size
    if args.ppo_epochs is not None:
        update["ppo_epochs"] = args.ppo_epochs
    if args.lora_rank is not None:
        update["lora"] = config.lora.model_copy(update={"rank": args.lora_rank})
    if args.no_critic_warmup:
        update["critic_warmup"] = config.critic_warmup.model_copy(update={"enabled": False})
    if args.allow_fresh_lora:
        update["allow_fresh_lora"] = True

    if not update:
        return config
    return config.model_copy(update=update)


def _validate_task_ids(config: PPOTrainConfig) -> None:
    """Refuse to launch with bogus task IDs — discovering this on a GPU is expensive."""

    from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS

    valid = set(SOLAR_TASKS.keys() if isinstance(SOLAR_TASKS, dict) else SOLAR_TASKS)
    requested: set[str] = set()
    if config.train_tasks:
        requested.update(config.train_tasks)
    if config.eval.held_out_tasks:
        requested.update(config.eval.held_out_tasks)
    unknown = sorted(task for task in requested if task not in valid)
    if unknown:
        raise SystemExit(
            "PPO config references unknown task ids: "
            f"{unknown}. Valid task ids: {sorted(valid)[:5]}... ({len(valid)} total)"
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    configure_gpu_runtime_env()
    load_dotenv_if_present(REPO_ROOT / ".env")
    args = parse_args()
    config = _apply_cli_overrides(load_config(args.config), args)
    _validate_task_ids(config)

    LOG.info(
        "config: model=%s sft=%s fresh_lora=%s steps=%d rollouts/step=%d kl=%.4f actor_lr=%g critic_lr=%g",
        config.model_name,
        config.sft_checkpoint or "(none)",
        config.allow_fresh_lora,
        config.total_steps,
        config.rollout.rollout_batch_size,
        config.algorithm.kl_coef,
        config.optimizer.actor_lr,
        config.optimizer.critic_lr,
    )
    LOG.info("env: hf_token_visible=%s names=%s", hf_token_visible(), visible_token_names())

    if args.dry_run:
        # Resolve which tasks would be trained on, without loading the model.
        from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS

        held_out = set(config.eval.held_out_tasks)
        if config.train_tasks is not None:
            train_tasks = list(config.train_tasks)
        else:
            train_tasks = [task_id for task_id in SOLAR_TASKS if task_id not in held_out]
        print(json.dumps({
            "config": config.model_dump(),
            "n_train_tasks": len(train_tasks),
            "train_tasks_sample": train_tasks[:5],
            "held_out_tasks": sorted(held_out),
        }, indent=2, sort_keys=True))
        return 0

    # Lazy import — keeps --help fast and lets dry-run skip torch entirely.
    from training.ppo.trainer import PPOTrainer

    trainer = PPOTrainer(config)
    if args.no_vllm:
        LOG.info("--no-vllm: validating model load only, skipping fit()")
        trainer.save_checkpoint(0, label="bootstrap")
        return 0
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
