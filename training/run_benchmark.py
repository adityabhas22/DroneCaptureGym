"""Run the cross-policy regression benchmark and emit JSON + Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dronecaptureops.evaluation.benchmark import DEFAULT_POLICIES, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="DroneCaptureOps regression benchmark across suites and task IDs.")
    parser.add_argument(
        "--policies",
        default=",".join(DEFAULT_POLICIES),
        help="Comma-separated policy names (random,weak_scripted,scripted).",
    )
    parser.add_argument("--suites", default="", help="Comma-separated suite names; empty = all.")
    parser.add_argument("--tasks", default="", help="Comma-separated task IDs; empty = all SOLAR_TASKS.")
    parser.add_argument("--task-seeds", default="2101,2202,2301", help="Comma-separated seeds for task fan-out.")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    parser.add_argument("--markdown", default="", help="Optional Markdown summary path.")
    args = parser.parse_args()

    report = run_benchmark(
        policy_names=tuple(name.strip() for name in args.policies.split(",") if name.strip()),
        suite_names=tuple(name.strip() for name in args.suites.split(",") if name.strip()) or None,
        task_ids=tuple(name.strip() for name in args.tasks.split(",") if name.strip()) or None,
        task_seeds=tuple(int(seed) for seed in args.task_seeds.split(",") if seed.strip()),
    )
    print(report.to_markdown())
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report.to_markdown(), encoding="utf-8")


if __name__ == "__main__":
    main()
