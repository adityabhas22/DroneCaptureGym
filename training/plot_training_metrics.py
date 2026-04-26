"""Plot real SFT trainer metrics for submission artifacts.

The script accepts either a raw trainer log-history JSON list or a
Hugging Face `trainer_state.json` file containing `log_history`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"trainer log not found: {path}")
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("log_history"), list):
        return payload["log_history"]
    raise SystemExit(f"unsupported trainer log format: {path}")


def _series(rows: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        if key in row:
            xs.append(int(row.get("step", len(xs))))
            ys.append(float(row[key]))
    return xs, ys


def plot_training_metrics(trainer_log: Path, output_dir: Path) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on training extras
        raise SystemExit("plotting requires matplotlib; install with `pip install -e '.[train]'`.") from exc

    history = _load_history(trainer_log)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [row for row in history if "loss" in row]
    eval_rows = [row for row in history if "eval_loss" in row]
    train_steps, train_loss = _series(train_rows, "loss")
    eval_steps, eval_loss = _series(eval_rows, "eval_loss")
    lr_steps, learning_rate = _series(train_rows, "learning_rate")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(train_steps, train_loss, marker="o", label="train_loss")
    if eval_steps:
        axes[0].plot(eval_steps, eval_loss, marker="s", label="eval_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if lr_steps:
        axes[1].plot(lr_steps, learning_rate, marker="o", color="#2ca02c")
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("lr")
    axes[1].grid(True, alpha=0.3)

    out_path = output_dir / "training_metrics.png"
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot DroneCaptureOps training metrics from a real trainer log.")
    parser.add_argument("--trainer-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/submission-plots"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot_training_metrics(args.trainer_log, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
