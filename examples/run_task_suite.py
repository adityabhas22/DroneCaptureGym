"""Run deterministic scripted rollouts for every solar mission task.

This module used to host its own procedural solver. That logic now lives in
`dronecaptureops.agent.oracle.TaskOraclePolicy`, which the rest of the
training pipeline (SFT data generation, eval harness, etc.) consumes.

We keep `solve_task` as a thin compatibility shim so existing examples and
tests that import it still work — it just delegates to `RolloutRunner`
running `TaskOraclePolicy`. The end-of-episode observation is returned for
backward compatibility.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dronecaptureops.agent import RolloutRunner, TaskOraclePolicy
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import DroneObservation
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS, get_solar_task


def solve_task(task_id: str, seed: int = 7) -> DroneObservation:
    """Run the canonical oracle on one task; return the final observation.

    Kept as a function-level entry point for legacy callers. New code should
    use `TaskOraclePolicy` directly through `RolloutRunner`.
    """

    spec = get_solar_task(task_id)
    runner = RolloutRunner(env=DroneCaptureOpsEnvironment())
    result = runner.run(
        TaskOraclePolicy(task_id=task_id),
        seed=seed,
        task_id=task_id,
        max_steps=spec.max_steps,
    )
    return DroneObservation.model_validate(result.final_observation)


def main() -> None:
    """Solve every task in the catalog and print a summary."""

    runner = RolloutRunner()
    results: dict[str, dict] = {}
    for task_id, spec in SOLAR_TASKS.items():
        result = runner.run(
            TaskOraclePolicy(task_id=task_id),
            seed=7,
            task_id=task_id,
            max_steps=spec.max_steps,
        )
        final = result.final_observation
        checklist = final.get("checklist_status", {})
        telemetry = final.get("telemetry", {})
        results[task_id] = {
            "reward": result.total_reward,
            "done": bool(final.get("done")),
            "complete": bool(checklist.get("complete")),
            "battery_pct": telemetry.get("battery", {}).get("level_pct"),
            "warnings": result.final_observation.get("warnings", []),
            "steps": result.steps,
        }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
