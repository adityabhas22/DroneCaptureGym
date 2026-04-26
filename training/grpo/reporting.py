"""Thin GRPO wrapper around the existing PPO reporting helpers.

Most of the reporting code is trainer-agnostic: it iterates rollouts and
metric rows from JSONL files. We re-export the core helpers and add
GRPO-flavoured per-step Markdown so the language matches what the user
sees in the runbook.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from training.ppo.reporting import (
    METRIC_SERIES,
    _compact_tool_calls,
    _fmt,
    _markdown_table,
    append_jsonl,
    build_training_report as _build_training_report_ppo,
    rollout_record,
)


GRPO_METRIC_SERIES = (
    ("mean_total_reward", "Mean Reward"),
    ("success_rate", "Success Rate"),
    ("parse_error_rate", "Parse Error Rate"),
    ("kl_to_ref", "KL To Ref"),
    ("policy_loss", "Policy Loss"),
    ("entropy", "Entropy"),
    ("grad_norm", "Grad Norm"),
    ("advantage_mean", "Advantage Mean"),
    ("advantage_std", "Advantage Std"),
)


def write_step_report(
    output_dir: Path,
    *,
    step: int,
    metrics: dict[str, Any],
    rollouts: list[dict[str, Any]],
) -> Path:
    """Per-step Markdown summary tailored to GRPO scalar series."""

    report_dir = output_dir / "reports" / "steps"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"step_{step:04d}.md"

    failure_counts = Counter(str(row.get("failure_mode", "unknown")) for row in rollouts)
    by_group: dict[int, list[dict[str, Any]]] = {}
    for row in rollouts:
        gp = int(row.get("prompt_index", 0))
        by_group.setdefault(gp, []).append(row)

    lines = [
        f"# GRPO Step {step} Report",
        "",
        "## Scalar Metrics",
        "",
        _markdown_table(
            ["metric", "value"],
            [(key, _fmt(metrics.get(key))) for key, _label in GRPO_METRIC_SERIES if key in metrics],
        ),
        "",
        "## Rollouts",
        "",
        _markdown_table(
            ["prompt", "group", "task", "seed", "reward", "success", "failure", "steps"],
            [
                (
                    row.get("prompt_index"),
                    row.get("group_index"),
                    row.get("task_id"),
                    row.get("seed"),
                    _fmt(row.get("total_reward")),
                    row.get("success"),
                    row.get("failure_mode"),
                    row.get("steps"),
                )
                for row in rollouts
            ],
        ),
        "",
        "## Per-Group Reward Spread",
        "",
    ]
    if by_group:
        rows = []
        for prompt_idx, items in sorted(by_group.items()):
            rewards = [float(row.get("total_reward", 0.0) or 0.0) for row in items]
            rows.append(
                (
                    prompt_idx,
                    items[0].get("task_id"),
                    len(items),
                    _fmt(min(rewards)),
                    _fmt(max(rewards)),
                    _fmt(sum(rewards) / max(len(rewards), 1)),
                )
            )
        lines.append(_markdown_table(["prompt", "task", "G", "min", "max", "mean"], rows))
    else:
        lines.append("No grouped rollouts captured.")
    lines.extend(["", "## Failure Modes", ""])
    if failure_counts:
        total = sum(failure_counts.values())
        lines.append(
            _markdown_table(
                ["failure_mode", "count", "rate"],
                [(mode, count, _fmt(count / total)) for mode, count in failure_counts.most_common()],
            )
        )
    else:
        lines.append("No rollout-level diagnostics were captured.")
    lines.extend(["", "## What Happened", ""])
    for row in rollouts:
        lines.append(
            f"- prompt {row.get('prompt_index')} g{row.get('group_index')} on `{row.get('task_id')}`: "
            f"reward `{_fmt(row.get('total_reward'))}`, failure `{row.get('failure_mode')}`, "
            f"tools `{_compact_tool_calls(row.get('tool_calls') or {})}`."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_training_report(output_dir: str | Path) -> dict[str, Path]:
    """Reuse the PPO training report (it walks JSONL agnostically)."""

    return _build_training_report_ppo(output_dir)


__all__ = [
    "GRPO_METRIC_SERIES",
    "METRIC_SERIES",
    "append_jsonl",
    "build_training_report",
    "rollout_record",
    "write_step_report",
]
