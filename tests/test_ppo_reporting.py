from __future__ import annotations

import json
from pathlib import Path

from training.ppo.reporting import build_training_report


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_training_report_writes_markdown_and_svg_charts(tmp_path: Path):
    _write_jsonl(
        tmp_path / "metrics.jsonl",
        [
            {
                "step": 1,
                "mean_total_reward": 0.0,
                "success_rate": 0.0,
                "parse_error_rate": 0.0,
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "kl_to_ref": 0.5,
                "rollout_secs": 1.0,
                "forward_secs": 0.5,
                "update_secs": 0.75,
            },
            {
                "step": 2,
                "mean_total_reward": 0.25,
                "success_rate": 0.5,
                "parse_error_rate": 0.0,
                "policy_loss": 0.08,
                "value_loss": 0.1,
                "kl_to_ref": 0.6,
                "rollout_secs": 1.1,
                "forward_secs": 0.4,
                "update_secs": 0.7,
            },
        ],
    )
    _write_jsonl(
        tmp_path / "rollouts" / "rollouts.jsonl",
        [
            {
                "task_id": "basic_thermal_survey",
                "success": False,
                "total_reward": 0.0,
                "parse_errors": 0,
                "failure_mode": "no_submit",
                "reward_components": {"process_reward": 0.1, "total": 0.0},
            },
            {
                "task_id": "basic_thermal_survey",
                "success": True,
                "total_reward": 0.5,
                "parse_errors": 0,
                "failure_mode": "success",
                "reward_components": {"process_reward": 0.2, "total": 0.5},
            },
        ],
    )

    paths = build_training_report(tmp_path)

    assert paths["training_report"].exists()
    assert "Latest mean reward" in paths["training_report"].read_text(encoding="utf-8")
    assert paths["reward_success_chart"].read_text(encoding="utf-8").startswith("<svg")
    assert paths["failure_modes_chart"].exists()
