"""Run a named scenario suite with a baseline policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dronecaptureops.evaluation.policies import get_policy
from dronecaptureops.evaluation.suite_runner import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a DroneCaptureOps scenario suite.")
    parser.add_argument("--suite", default="smoke", help="Suite name such as smoke, demo, hard_eval")
    parser.add_argument("--policy", default="scripted", choices=["random", "weak_scripted", "scripted"])
    parser.add_argument("--include-rollouts", action="store_true", help="Include full trajectories in JSON output")
    parser.add_argument("--output", default="", help="Optional JSON report path")
    args = parser.parse_args()

    report = run_suite(
        get_policy(args.policy),
        suite=args.suite,
        include_rollouts=args.include_rollouts,
    )
    print(report.to_markdown())
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
