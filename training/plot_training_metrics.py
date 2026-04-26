"""Plot real training/evaluation metrics for submission evidence.

This script intentionally refuses to synthesize data. Point it at real
`trainer_state.json`, `metrics/trainer_log_history.json`, or evaluation JSONL
files produced by the training/evaluation commands.

Examples:
    python -m training.plot_training_metrics \
        --trainer-log artifacts/sft-checkpoints/metrics/trainer_log_history.json \
        --output-dir artifacts/submission-plots

    python -m training.plot_training_metrics \
        --trainer-state artifacts/sft-checkpoints/checkpoint-50/trainer_state.json \
        --eval-jsonl artifacts/eval/model-eval.jsonl \
        --output-dir artifacts/submission-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    if not path.exists():
        raise SystemExit(f"metrics file not found: {path}")
    return json.loads(path.read_text())


def _load_log_history(*, trainer_state: Path | None, trainer_log: Path | None) -> list[dict[str, Any]]:
    if trainer_log:
        payload = _load_json(trainer_log)
        if not isinstance(payload, list):
            raise SystemExit(f"expected list log history in {trainer_log}")
        return [row for row in payload if isinstance(row, dict)]

    if trainer_state:
        payload = _load_json(trainer_state)
        history = payload.get("log_history") if isinstance(payload, dict) else None
        if not isinstance(history, list):
            raise SystemExit(f"expected log_history list in {trainer_state}")
        return [row for row in history if isinstance(row, dict)]

    raise SystemExit("provide --trainer-log or --trainer-state")


def _load_eval_rewards(path: Path | None) -> tuple[list[int], list[float]]:
    if path is None:
        return [], []
    if not path.exists():
        raise SystemExit(f"eval JSONL not found: {path}")

    steps: list[int] = []
    rewards: list[float] = []
    for idx, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        reward = row.get("total_reward", row.get("reward", row.get("mean_reward")))
        if reward is None:
            continue
        steps.append(int(row.get("step", row.get("global_step", idx))))
        rewards.append(float(reward))
    return steps, rewards


def _series(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for idx, row in enumerate(history, start=1):
        if key not in row:
            continue
        step = int(row.get("step", row.get("global_step", idx)))
        xs.append(step)
        ys.append(float(row[key]))
    return xs, ys


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot real DroneCaptureOps training metrics.")
    parser.add_argument("--trainer-state", type=Path, default=None, help="Path to Hugging Face trainer_state.json.")
    parser.add_argument("--trainer-log", type=Path, default=None, help="Path to metrics/trainer_log_history.json.")
    parser.add_argument("--eval-jsonl", type=Path, default=None, help="Optional JSONL from training/eval_models.py.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/submission-plots"))
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install training extras first: pip install -e '.[train]'") from exc

    history = _load_log_history(trainer_state=args.trainer_state, trainer_log=args.trainer_log)
    train_steps, train_loss = _series(history, "loss")
    eval_steps, eval_loss = _series(history, "eval_loss")
    reward_steps, rewards = _load_eval_rewards(args.eval_jsonl)

    if not train_loss and not eval_loss and not rewards:
        raise SystemExit("no loss/eval_loss/reward data found; refusing to generate empty evidence plots")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if train_loss or eval_loss:
        plt.figure(figsize=(8, 5))
        if train_loss:
            plt.plot(train_steps, train_loss, label="train_loss")
        if eval_loss:
            plt.plot(eval_steps, eval_loss, label="eval_loss")
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("DroneCaptureOps Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "loss_curve.png", dpi=160)
        plt.close()

    if rewards:
        plt.figure(figsize=(8, 5))
        plt.plot(reward_steps, rewards, marker="o", label="eval_reward")
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.title("DroneCaptureOps Evaluation Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.output_dir / "reward_curve.png", dpi=160)
        plt.close()

    print(f"wrote plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

