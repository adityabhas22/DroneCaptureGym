"""GRPO trainer entrypoint — no-vLLM, HF generate-only.

Mirrors the PPO entrypoint's CLI shape so HF Jobs and the local dev
loop can drive both the same way::

    python -m training.train_grpo                                       # built-in defaults
    python -m training.train_grpo --config training/configs/grpo_tiny_4b_l40_kl005.yaml
    python -m training.train_grpo --model Qwen/Qwen3-4B-Instruct-2507 --output-dir artifacts/grpo

Heavy imports (torch / transformers / peft) are deferred into
``GRPOTrainer.__init__`` so ``--dry-run`` and ``--help`` stay fast.
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
from training.grpo.config import GRPOTrainConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "grpo_tiny_4b_l40.yaml"

LOG = logging.getLogger("dronecaptureops.grpo.train")


def configure_gpu_runtime_env() -> None:
    """Set CUDA safety env vars before importing torch.

    GRPO does not start vLLM at all on this branch, so we drop the
    ``VLLM_WORKER_MULTIPROC_METHOD`` setting; only the NVML cuda check
    remains useful so PyTorch reports CUDA availability correctly even
    when fork+CUDA was previously touched.
    """

    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")


def load_config(path: Path | None) -> GRPOTrainConfig:
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        LOG.warning("config path %s not found, using built-in defaults", path)
        return GRPOTrainConfig()
    raw = yaml.safe_load(path.read_text()) or {}
    try:
        return GRPOTrainConfig.model_validate(raw)
    except ValidationError as exc:
        raise SystemExit(f"invalid GRPO trainer config: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO trainer for DroneCaptureOps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--model", dest="model_name", default=None, help="Override base model.")
    parser.add_argument("--sft-checkpoint", default=None, help="Override SFT LoRA path.")
    parser.add_argument("--output-dir", default=None, help="Override output directory.")
    parser.add_argument("--total-steps", type=int, default=None, help="Override total GRPO steps.")
    parser.add_argument("--prompts-per-step", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--grpo-epochs", type=int, default=None)
    parser.add_argument("--actor-lr", type=float, default=None)
    parser.add_argument("--kl-coef", type=float, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--allow-fresh-lora",
        action="store_true",
        help="Allow GRPO to start from a fresh LoRA when no SFT checkpoint is loaded.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and resolve task list without launching training.",
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Build the trainer (loads model + SFT LoRA on GPU) but skip fit() — used for smoke checks.",
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


def _apply_cli_overrides(config: GRPOTrainConfig, args: argparse.Namespace) -> GRPOTrainConfig:
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

    if args.actor_lr is not None:
        update["optimizer"] = config.optimizer.model_copy(update={"actor_lr": args.actor_lr})
    if args.kl_coef is not None:
        update["algorithm"] = config.algorithm.model_copy(update={"kl_coef": args.kl_coef})
    rollout_updates: dict[str, Any] = {}
    if args.prompts_per_step is not None:
        rollout_updates["prompts_per_step"] = args.prompts_per_step
    if args.group_size is not None:
        rollout_updates["group_size"] = args.group_size
    if rollout_updates:
        update["rollout"] = config.rollout.model_copy(update=rollout_updates)
    if args.minibatch_size is not None:
        update["minibatch_size"] = args.minibatch_size
    if args.grpo_epochs is not None:
        update["grpo_epochs"] = args.grpo_epochs
    if args.lora_rank is not None:
        update["lora"] = config.lora.model_copy(update={"rank": args.lora_rank})
    if args.allow_fresh_lora:
        update["allow_fresh_lora"] = True

    if not update:
        return config
    return config.model_copy(update=update)


def _validate_task_ids(config: GRPOTrainConfig) -> None:
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
            "GRPO config references unknown task ids: "
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
        "config: model=%s sft=%s fresh_lora=%s steps=%d prompts/step=%d group=%d kl=%.4f actor_lr=%g",
        config.model_name,
        config.sft_checkpoint or "(none)",
        config.allow_fresh_lora,
        config.total_steps,
        config.rollout.prompts_per_step,
        config.rollout.group_size,
        config.algorithm.kl_coef,
        config.optimizer.actor_lr,
    )
    LOG.info("env: hf_token_visible=%s names=%s", hf_token_visible(), visible_token_names())

    if args.dry_run:
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

    from training.grpo.trainer import GRPOTrainer

    trainer = GRPOTrainer(config)
    if args.no_fit:
        LOG.info("--no-fit: validating model load only, skipping fit()")
        trainer.save_checkpoint(0, label="bootstrap")
        return 0
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
