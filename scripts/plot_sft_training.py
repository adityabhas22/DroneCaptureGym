"""Plot SFT training curves from a HuggingFace `trainer_state.json`.

Reads the log_history written by TRL's SFTTrainer, separates train/eval
rows, and renders a six-panel summary (loss, eval gap, token accuracy,
LR schedule, grad norm, entropy) plus a small text annotation block.

Usage:
    python scripts/plot_sft_training.py \
        --state artifacts/sft/logs/last-checkpoint/trainer_state.json \
        --out artifacts/sft/logs/sft_training_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _split_history(log_history: list[dict]) -> tuple[list[dict], list[dict]]:
    train, evals = [], []
    for row in log_history:
        if "eval_loss" in row:
            evals.append(row)
        elif "loss" in row:
            train.append(row)
    return train, evals


def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in rows:
        if key in r:
            xs.append(r["step"])
            ys.append(r[key])
    return xs, ys


def plot(state_path: Path, out_path: Path) -> None:
    state = json.loads(state_path.read_text())
    train, evals = _split_history(state["log_history"])

    train_steps, train_loss = _series(train, "loss")
    eval_steps, eval_loss = _series(evals, "eval_loss")
    _, train_acc = _series(train, "mean_token_accuracy")
    _, eval_acc = _series(evals, "eval_mean_token_accuracy")
    _, train_lr = _series(train, "learning_rate")
    _, train_grad = _series(train, "grad_norm")
    _, train_ent = _series(train, "entropy")
    _, eval_ent = _series(evals, "eval_entropy")
    _, train_tokens = _series(train, "num_tokens")

    max_steps = state.get("max_steps")
    best_step = state.get("best_global_step")
    best_metric = state.get("best_metric")
    last_step = state.get("global_step")
    epochs_seen = state.get("epoch")
    total_flos = state.get("total_flos")
    tokens_total = train_tokens[-1] if train_tokens else 0

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    # 1. Loss
    ax = axes[0, 0]
    ax.plot(train_steps, train_loss, marker="o", label="train", color="#1f77b4")
    ax.plot(eval_steps, eval_loss, marker="s", label="eval", color="#d62728")
    if best_step is not None:
        ax.axvline(best_step, color="gray", ls="--", lw=1)
        ax.annotate(
            f"best eval={best_metric:.3f}\n@step {best_step}",
            xy=(best_step, best_metric),
            xytext=(8, 12), textcoords="offset points",
            fontsize=8, color="gray",
        )
    ax.set_title("Loss"); ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3); ax.legend()

    # 2. Eval-train gap (overfit watch)
    ax = axes[0, 1]
    if eval_steps:
        train_at_eval = np.interp(eval_steps, train_steps, train_loss)
        gap = np.array(eval_loss) - train_at_eval
        ax.plot(eval_steps, gap, marker="s", color="#9467bd")
        ax.axhline(0, color="black", lw=0.7)
        for s, g in zip(eval_steps, gap):
            ax.annotate(f"{g:+.2f}", xy=(s, g), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8)
    ax.set_title("Eval − Train loss gap")
    ax.set_xlabel("step"); ax.set_ylabel("Δ loss")
    ax.grid(True, alpha=0.3)

    # 3. Token accuracy
    ax = axes[0, 2]
    ax.plot(train_steps, train_acc, marker="o", label="train", color="#1f77b4")
    ax.plot(eval_steps, eval_acc, marker="s", label="eval", color="#d62728")
    ax.set_title("Mean token accuracy (assistant-only)")
    ax.set_xlabel("step"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.85, 0.95); ax.grid(True, alpha=0.3); ax.legend()

    # 4. LR schedule
    ax = axes[1, 0]
    ax.plot(train_steps, train_lr, marker="o", color="#2ca02c")
    if max_steps:
        ax.axvspan(0, max(train_steps[0], 1), alpha=0.08, color="orange",
                   label="warmup region")
        ax.axvline(max_steps, color="black", ls=":", lw=1, label=f"max_steps={max_steps}")
        ax.legend(fontsize=8)
    ax.set_title("Learning rate (cosine schedule)")
    ax.set_xlabel("step"); ax.set_ylabel("lr"); ax.grid(True, alpha=0.3)

    # 5. Grad norm
    ax = axes[1, 1]
    ax.plot(train_steps, train_grad, marker="o", color="#ff7f0e")
    ax.set_title("Gradient norm")
    ax.set_xlabel("step"); ax.set_ylabel("‖g‖"); ax.grid(True, alpha=0.3)

    # 6. Entropy
    ax = axes[1, 2]
    ax.plot(train_steps, train_ent, marker="o", label="train", color="#1f77b4")
    ax.plot(eval_steps, eval_ent, marker="s", label="eval", color="#d62728")
    ax.set_title("Token entropy (lower = sharper distribution)")
    ax.set_xlabel("step"); ax.set_ylabel("entropy (nats)")
    ax.grid(True, alpha=0.3); ax.legend()

    title_lines = [
        "Qwen3-4B-Instruct  •  LoRA SFT (DroneCaptureOps)",
        f"reached step {last_step}/{max_steps}  ({epochs_seen:.2f} epochs)  "
        f"•  best eval_loss = {best_metric:.4f} @ step {best_step}  "
        f"•  tokens seen = {tokens_total/1e6:.2f}M  "
        f"•  total FLOPs = {total_flos:.2e}",
    ]
    fig.suptitle("\n".join(title_lines), fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state",
        type=Path,
        default=repo_root / "artifacts/sft/logs/last-checkpoint/trainer_state.json",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=repo_root / "artifacts/sft/logs/sft_training_curves.png",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot(args.state, args.out)
