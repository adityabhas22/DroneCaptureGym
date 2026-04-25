"""Spec-aware scripted policy — wraps the all-45-task solver as a Policy.

The solver in `examples/run_task_suite.py` already encodes spec-driven
behaviour for every task in `SOLAR_TASKS`. This module adapts it to the
shared `Policy` protocol so the SFT data generator and any other consumer
of `RolloutRunner` can drive episodes through the same interface they use
for `OpenAIChatPolicy`, `HFInferencePolicy`, etc.

How it works:
    Each `SpecAwareScriptedPolicy(task_id=..., seed=...)` runs the solver
    once, in a private env, on first `next_action`. The action sequence
    that solver emits is captured and replayed action-by-action against
    the runner's env. Because the env is deterministic given (task, seed),
    the trajectory in the runner's env mirrors the private one — same
    observations, same outcomes — but the rollout is recorded by the
    runner's standard machinery (SFT data generator, eval harness, etc).

If the captured sequence runs out before the env finishes (rare; usually
means the runner is using a smaller `max_steps` than the solver expected),
the policy falls back to `hover` so it never raises a parse error mid-
episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dronecaptureops.agent.policies import AgentContext
from dronecaptureops.core.models import DroneObservation, RawDroneAction


@dataclass
class SpecAwareScriptedPolicy:
    """Replay the spec-aware solver's actions one-per-step.

    Pass `task_id` so the policy knows which task to solve. `seed` selects
    the env world; `strategy` ∈ {0, 1, 2} selects between alternate valid
    solution paths so the SFT corpus has multiple action sequences per task
    rather than one memorisable template.
    """

    task_id: str
    seed: int = 0
    strategy: int = 0
    name: str = "spec_aware_scripted"
    _actions: list[RawDroneAction] = field(default_factory=list, init=False)
    _cursor: int = field(default=0, init=False)
    _initialised: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # Encode strategy in the policy name so generated trajectories are
        # tagged with which solution path produced them (visible in SFT JSONL).
        self.name = f"spec_aware_scripted_s{self.strategy}"

    def _initialise(self) -> None:
        # Local import to avoid a hard dependency on `examples/` at import time.
        from examples.run_task_suite import solve_task_actions

        self._actions, _ = solve_task_actions(
            self.task_id, seed=self.seed, strategy=self.strategy
        )
        self._initialised = True

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        if not self._initialised:
            self._initialise()
        if self._cursor < len(self._actions):
            action = self._actions[self._cursor]
            self._cursor += 1
            return action
        # Fallback — the solver finished but the runner is still asking.
        return RawDroneAction(tool_name="hover", arguments={"seconds": 1})


__all__ = ["SpecAwareScriptedPolicy"]
