"""Generate trajectory trace artifacts for one suite episode."""

from __future__ import annotations

import argparse

from dronecaptureops.evaluation.policies import get_policy
from dronecaptureops.evaluation.rollout import RolloutRunner
from dronecaptureops.evaluation.tracing import write_trace_artifacts
from dronecaptureops.generation.suites import get_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace one DroneCaptureOps suite episode.")
    parser.add_argument("--suite", default="demo")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--policy", default="scripted", choices=["random", "weak_scripted", "scripted"])
    parser.add_argument("--output-dir", default="artifacts/trace")
    args = parser.parse_args()

    suite = get_suite(args.suite)
    episode = suite.episodes[args.episode_index]
    rollout = RolloutRunner().run(
        get_policy(args.policy),
        seed=episode.seed,
        scenario_family=episode.scenario_family,
        task_id=episode.task_id or None,
        max_steps=episode.max_steps,
    )
    paths = write_trace_artifacts(rollout, args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
