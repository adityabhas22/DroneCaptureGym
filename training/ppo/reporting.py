"""Reporting helpers for PPO training runs.

The trainer writes JSONL so long runs are append-only and resilient to
interruptions. This module turns those rows into Markdown and SVG artifacts
that can be uploaded with HF Job outputs without adding plotting dependencies.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from dronecaptureops.agent.eval_metrics import trajectory_metrics


METRIC_SERIES = (
    ("mean_total_reward", "Mean Reward"),
    ("success_rate", "Success Rate"),
    ("parse_error_rate", "Parse Error Rate"),
    ("kl_to_ref", "KL To Ref"),
    ("policy_loss", "Policy Loss"),
    ("value_loss", "Value Loss"),
    ("entropy", "Entropy"),
    ("grad_norm", "Grad Norm"),
)


def rollout_record(output: Any, *, step: int, rollout_index: int, include_messages: bool = False) -> dict[str, Any]:
    """Return a compact, JSON-serializable diagnostic row for one rollout."""

    result = output.result
    diagnostics = trajectory_metrics(result)
    trajectory = list(result.trajectory or [])
    parse_errors = sum(1 for item in trajectory if item.parse_error is not None)
    actions = [
        {
            "step": item.step,
            "tool_name": item.action.get("tool_name") if item.parse_error is None else "_parse_error",
            "reward": float(item.reward),
            "done": bool(item.done),
            "parse_error": str(item.parse_error) if item.parse_error is not None else None,
        }
        for item in trajectory
    ]
    record: dict[str, Any] = {
        "step": int(step),
        "rollout_index": int(rollout_index),
        "task_id": output.spec.task_id,
        "scenario_family": output.spec.scenario_family or result.scenario_family,
        "seed": int(output.spec.seed),
        "policy_name": result.policy_name,
        "episode_id": result.episode_id,
        "success": bool(result.success),
        "done": bool(trajectory[-1].done) if trajectory else False,
        "steps": int(result.steps),
        "total_reward": float(result.total_reward),
        "parse_errors": int(parse_errors),
        "failure_mode": diagnostics.failure_mode,
        "checkpoints": diagnostics.checkpoints,
        "coverage": diagnostics.coverage,
        "safety": diagnostics.safety,
        "reward_components": diagnostics.reward_components,
        "tool_calls": diagnostics.tool_calls,
        "actions": actions,
    }
    if include_messages:
        record["messages"] = output.messages
    return record


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_step_report(output_dir: Path, *, step: int, metrics: dict[str, Any], rollouts: list[dict[str, Any]]) -> Path:
    report_dir = output_dir / "reports" / "steps"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"step_{step:04d}.md"
    lines = [
        f"# PPO Step {step} Report",
        "",
        "## Scalar Metrics",
        "",
        _markdown_table(
            ["metric", "value"],
            [(key, _fmt(metrics.get(key))) for key, _label in METRIC_SERIES if key in metrics],
        ),
        "",
        "## Rollouts",
        "",
        _markdown_table(
            ["idx", "task", "seed", "reward", "success", "failure", "steps", "parse_errors"],
            [
                (
                    row["rollout_index"],
                    row.get("task_id"),
                    row.get("seed"),
                    _fmt(row.get("total_reward")),
                    row.get("success"),
                    row.get("failure_mode"),
                    row.get("steps"),
                    row.get("parse_errors"),
                )
                for row in rollouts
            ],
        ),
        "",
        "## What Happened",
        "",
    ]
    if not rollouts:
        lines.append("- No rollout diagnostics were captured for this step.")
    for row in rollouts:
        lines.append(
            f"- Rollout {row['rollout_index']} on `{row.get('task_id')}`: "
            f"reward `{_fmt(row.get('total_reward'))}`, failure `{row.get('failure_mode')}`, "
            f"tools `{_compact_tool_calls(row.get('tool_calls') or {})}`."
        )
        coverage = row.get("coverage") or {}
        missing = coverage.get("missing_rows") or []
        if missing:
            lines.append(f"  Missing rows: `{', '.join(str(item) for item in missing)}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_training_report(output_dir: str | Path) -> dict[str, Path]:
    """Build Markdown and SVG summary artifacts from a PPO output directory."""

    out = Path(output_dir)
    reports = out / "reports"
    charts = reports / "charts"
    reports.mkdir(parents=True, exist_ok=True)
    charts.mkdir(parents=True, exist_ok=True)

    metrics = _read_jsonl(out / "metrics.jsonl")
    rollouts = _read_jsonl(out / "rollouts" / "rollouts.jsonl")
    paths: dict[str, Path] = {}
    paths["training_report"] = _write_training_markdown(reports / "training_report.md", metrics, rollouts)
    if metrics:
        paths["reward_success_chart"] = _write_line_chart(
            charts / "reward_success.svg",
            metrics,
            [("mean_total_reward", "reward"), ("success_rate", "success"), ("parse_error_rate", "parse errors")],
            title="Reward, Success, Parse Errors",
        )
        paths["loss_kl_chart"] = _write_line_chart(
            charts / "loss_kl.svg",
            metrics,
            [("policy_loss", "policy loss"), ("value_loss", "value loss"), ("kl_to_ref", "KL to ref")],
            title="Losses and KL",
        )
        paths["timing_chart"] = _write_line_chart(
            charts / "timing.svg",
            metrics,
            [("rollout_secs", "rollout"), ("forward_secs", "forward"), ("update_secs", "update")],
            title="Step Timing",
        )
    if rollouts:
        paths["failure_modes_chart"] = _write_bar_chart(
            charts / "failure_modes.svg",
            Counter(str(row.get("failure_mode", "unknown")) for row in rollouts),
            title="Failure Modes",
        )
        reward_component_means = _mean_nested(rollouts, "reward_components")
        if reward_component_means:
            paths["reward_components_chart"] = _write_bar_chart(
                charts / "reward_components.svg",
                reward_component_means,
                title="Mean Reward Components",
            )
    return paths


def _write_training_markdown(path: Path, metrics: list[dict[str, Any]], rollouts: list[dict[str, Any]]) -> Path:
    latest = metrics[-1] if metrics else {}
    failure_counts = Counter(str(row.get("failure_mode", "unknown")) for row in rollouts)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rollouts:
        by_task[str(row.get("task_id", "unknown"))].append(row)

    lines = [
        "# PPO Training Report",
        "",
        "## Executive Summary",
        "",
    ]
    if not metrics:
        lines.append("- No scalar metrics were found.")
    else:
        lines.extend(
            [
                f"- Steps logged: `{len(metrics)}`",
                f"- Latest mean reward: `{_fmt(latest.get('mean_total_reward'))}`",
                f"- Latest success rate: `{_fmt(latest.get('success_rate'))}`",
                f"- Latest parse error rate: `{_fmt(latest.get('parse_error_rate'))}`",
                f"- Latest KL to reference: `{_fmt(latest.get('kl_to_ref'))}`",
            ]
        )
    lines.extend(["", "## Latest Scalars", ""])
    lines.append(
        _markdown_table(
            ["metric", "value"],
            [(key, _fmt(latest.get(key))) for key, _label in METRIC_SERIES if key in latest],
        )
    )
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
        lines.append("No rollout-level diagnostics were found.")
    lines.extend(["", "## Task Breakdown", ""])
    if by_task:
        rows = []
        for task, items in sorted(by_task.items()):
            rows.append(
                (
                    task,
                    len(items),
                    _fmt(_mean(row.get("total_reward") for row in items)),
                    _fmt(sum(1 for row in items if row.get("success")) / len(items)),
                    _fmt(_mean(row.get("parse_errors") for row in items)),
                    Counter(str(row.get("failure_mode", "unknown")) for row in items).most_common(1)[0][0],
                )
            )
        lines.append(_markdown_table(["task", "n", "mean_reward", "success_rate", "mean_parse_errors", "top_failure"], rows))
    else:
        lines.append("No task-level rollout diagnostics were found.")
    lines.extend(
        [
            "",
            "## How To Read This",
            "",
            "- A one-step smoke only proves the training loop, checkpoint upload, and rollout/update mechanics.",
            "- Reward trends need multiple PPO steps and multiple rollouts per step before they mean anything.",
            "- `parse_error_rate=0` is the first gate; rising reward and stable KL are the next gates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_line_chart(path: Path, rows: list[dict[str, Any]], series: list[tuple[str, str]], *, title: str) -> Path:
    width, height = 900, 360
    margin = 52
    xs = [float(row.get("step", idx + 1)) for idx, row in enumerate(rows)]
    values = [float(row.get(key, 0.0) or 0.0) for row in rows for key, _label in series]
    y_min, y_max = _bounds(values)
    x_min, x_max = _bounds(xs)
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c"]
    parts = [_svg_header(width, height, title), _axes(width, height, margin, x_min, x_max, y_min, y_max)]
    for idx, (key, label) in enumerate(series):
        points = []
        for row_idx, row in enumerate(rows):
            x = _scale(float(row.get("step", row_idx + 1)), x_min, x_max, margin, width - margin)
            y = _scale(float(row.get(key, 0.0) or 0.0), y_min, y_max, height - margin, margin)
            points.append(f"{x:.1f},{y:.1f}")
        color = colors[idx % len(colors)]
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{" ".join(points)}" />')
        parts.append(f'<text x="{margin + idx * 170}" y="{height - 12}" fill="{color}" font-size="13">{_xml(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _write_bar_chart(path: Path, values: dict[str, float] | Counter[str], *, title: str) -> Path:
    width, height = 900, 420
    margin = 52
    items = list(values.items())
    items.sort(key=lambda item: float(item[1]), reverse=True)
    items = items[:12]
    max_value = max((float(value) for _key, value in items), default=1.0)
    bar_gap = 8
    bar_height = max(14, (height - 2 * margin) / max(len(items), 1) - bar_gap)
    parts = [_svg_header(width, height, title)]
    for idx, (key, value) in enumerate(items):
        y = margin + idx * (bar_height + bar_gap)
        bar_width = (float(value) / max_value) * (width - 360)
        parts.append(f'<text x="{margin}" y="{y + bar_height * 0.7:.1f}" font-size="12">{_xml(str(key)[:36])}</text>')
        parts.append(f'<rect x="300" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb" />')
        parts.append(f'<text x="{310 + bar_width:.1f}" y="{y + bar_height * 0.7:.1f}" font-size="12">{_fmt(value)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _svg_header(width: int, height: int, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white" />\n'
        f'<text x="24" y="30" font-size="20" font-family="Arial" font-weight="700">{_xml(title)}</text>'
    )


def _axes(width: int, height: int, margin: int, x_min: float, x_max: float, y_min: float, y_max: float) -> str:
    return "\n".join(
        [
            f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#111827" />',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#111827" />',
            f'<text x="{margin}" y="{height - 28}" font-size="11">step {_fmt(x_min)}</text>',
            f'<text x="{width - margin - 80}" y="{height - 28}" font-size="11">step {_fmt(x_max)}</text>',
            f'<text x="8" y="{height - margin}" font-size="11">{_fmt(y_min)}</text>',
            f'<text x="8" y="{margin + 4}" font-size="11">{_fmt(y_max)}</text>',
        ]
    )


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if abs(src_max - src_min) < 1e-12:
        return (dst_min + dst_max) / 2.0
    return dst_min + ((value - src_min) / (src_max - src_min)) * (dst_max - dst_min)


def _bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        pad = max(abs(low) * 0.1, 1.0)
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def _mean_nested(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        nested = row.get(key) or {}
        for name, value in nested.items():
            if isinstance(value, int | float):
                sums[name] += float(value)
                counts[name] += 1
    return {name: sums[name] / counts[name] for name in sums if counts[name]}


def _markdown_table(headers: list[str], rows: list[tuple[Any, ...]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _mean(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return sum(numeric) / len(numeric) if numeric else 0.0


def _fmt(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "n/a" if value is None else str(value)


def _compact_tool_calls(tool_calls: dict[str, Any]) -> str:
    if not tool_calls:
        return "none"
    return ", ".join(f"{tool}:{count}" for tool, count in sorted(tool_calls.items()))


def _xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build PPO Markdown/SVG reports from an output directory.")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    paths = build_training_report(args.output_dir)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
