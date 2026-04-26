"""vLLM rollout preflight for PPO configs.

This intentionally stops before PPO math. It proves the expensive/fragile
path first: load policy + LoRA, start vLLM, run one rollout, write a small
JSON diagnostic. Use this before launching one-step PPO smoke jobs.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from training.env_utils import hf_token_visible, load_dotenv_if_present, visible_token_names
from training.ppo.config import PPOTrainConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "ppo_smoke_1p7b_l4.yaml"
LOG = logging.getLogger("dronecaptureops.ppo.preflight_vllm")


def configure_gpu_runtime_env() -> None:
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight PPO vLLM rollout generation.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo-smoke/preflight-vllm.json"))
    parser.add_argument("--task", default=None, help="Override the first configured train task.")
    parser.add_argument("--seed", type=int, default=None, help="Override rollout seed.")
    return parser.parse_args()


def load_config(path: Path) -> PPOTrainConfig:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return PPOTrainConfig.model_validate(raw)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _base_record(config: PPOTrainConfig, *, config_path: Path, task_id: str, seed: int) -> dict[str, Any]:
    return {
        "config_path": str(config_path),
        "model_name": config.model_name,
        "sft_checkpoint": config.sft_checkpoint,
        "allow_fresh_lora": config.allow_fresh_lora,
        "task_id": task_id,
        "seed": seed,
        "hf_token_visible": hf_token_visible(),
        "hf_token_env_names": visible_token_names(),
        "vllm": {
            "max_model_len": config.vllm_max_model_len,
            "gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "dtype": config.vllm_dtype,
            "enforce_eager": config.vllm_enforce_eager,
            "worker_multiproc_method": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
            "flash_attn_version": os.environ.get("VLLM_FLASH_ATTN_VERSION"),
        },
        "rollout": config.rollout.model_dump(),
    }


def _summarize_rollout_result(result: Any) -> dict[str, Any]:
    parse_errors = sum(1 for step in result.trajectory if step.parse_error is not None)
    done = bool(result.trajectory[-1].done) if result.trajectory else False
    return {
        "steps": int(result.steps),
        "success": bool(result.success),
        "done": done,
        "total_reward": float(result.total_reward),
        "parse_errors": int(parse_errors),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    configure_gpu_runtime_env()
    load_dotenv_if_present(REPO_ROOT / ".env")
    args = parse_args()
    config = load_config(args.config)

    tasks = list(config.train_tasks or [])
    if not tasks and config.eval.held_out_tasks:
        raise SystemExit("preflight requires explicit train_tasks when eval held-outs are configured")
    task_id = args.task or (tasks[0] if tasks else "basic_thermal_survey")
    seed = args.seed if args.seed is not None else config.seed
    started = time.perf_counter()
    record = _base_record(config, config_path=args.config, task_id=task_id, seed=seed)

    try:
        from training.ppo.rollout_pool import PPORolloutSpec, run_rollout_batch
        from training.ppo.trainer import PPOTrainer

        trainer = PPOTrainer(config)
        lora_request = trainer._build_lora_request(step_id=0)  # noqa: SLF001 - preflight mirrors trainer lifecycle.
        engine = trainer._ensure_engine()  # noqa: SLF001
        outputs = run_rollout_batch(
            [PPORolloutSpec(task_id=task_id, seed=seed)],
            engine=engine,
            lora_request=lora_request,
            max_workers=1,
            temperature=config.rollout.temperature,
            top_p=config.rollout.top_p,
            max_tokens=config.rollout.max_new_tokens_per_turn,
            max_history_steps=config.rollout.max_history_steps,
            max_steps=config.rollout.max_episode_steps,
        )
        result = outputs[0].result
        record.update({
            "ok": True,
            "elapsed_secs": time.perf_counter() - started,
            **_summarize_rollout_result(result),
        })
        _write_json(args.output, record)
        LOG.info("preflight passed; wrote %s", args.output)
        return 0
    except Exception as exc:  # noqa: BLE001 - write diagnostics for failed jobs.
        record.update({
            "ok": False,
            "elapsed_secs": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        _write_json(args.output, record)
        LOG.exception("preflight failed; wrote %s", args.output)
        return 1


if __name__ == "__main__":
    sys.exit(main())
