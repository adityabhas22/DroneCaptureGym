"""Run a baseline suite evaluation."""

from __future__ import annotations

from dronecaptureops.evaluation.policies import get_policy
from dronecaptureops.evaluation.suite_runner import run_suite


def main() -> None:
    report = run_suite(get_policy("scripted"), suite="smoke")
    print(report.to_markdown())


if __name__ == "__main__":
    main()
