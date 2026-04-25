"""Concurrent rollout pool for PPO training.

Runs N rollouts of the existing `RolloutRunner` in parallel threads,
sharing a single `VLLMEngine`. Each thread holds its own
`DroneCaptureOpsEnvironment` (deterministic per seed) and a
`VLLMPolicy` instance — these aren't thread-safe across rollouts but
each thread owns its own copy.

vLLM's `generate()` releases the GIL while the actual inference runs
on the GPU and uses continuous batching across in-flight requests, so
multiple threads submitting concurrently get good throughput without
needing a multi-process server.

Outputs are PPO-ready: each rollout returns the trajectory rewards
(`RolloutResult`) plus the raw chat messages the policy actually
emitted (`VLLMPolicy.messages`). The trainer feeds the messages into
`tokenize_trajectory` and the rewards into `build_per_token_rewards`.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from dronecaptureops.agent.rollout import RolloutResult, RolloutRunner
from dronecaptureops.agent.vllm_policy import VLLMEngine, VLLMPolicy
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


LOG = logging.getLogger("dronecaptureops.ppo.rollout_pool")


@dataclass
class PPORolloutSpec:
    """One rollout request — task + seed + sampling params."""

    task_id: str | None
    seed: int
    scenario_family: str | None = None


@dataclass
class PPORolloutOutput:
    """Everything the trainer needs from one completed rollout."""

    spec: PPORolloutSpec
    result: RolloutResult                       # per-step rewards, parse errors
    messages: list[dict[str, Any]]              # raw chat including assistant text


def _run_one(
    spec: PPORolloutSpec,
    *,
    engine: VLLMEngine,
    lora_request: Any | None,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_history_steps: int,
    max_steps: int | None,
) -> PPORolloutOutput:
    """Run a single rollout end-to-end. Each thread calls this once."""

    env = DroneCaptureOpsEnvironment()
    runner = RolloutRunner(env=env)

    # `env.reset()` is called inside `runner.run()`; we must construct
    # the policy AFTER reset so the registry/world it captures match.
    # `RolloutRunner.run()` resets first — VLLMPolicy lazily fetches
    # registry/world on first `next_action`, which fires after reset.
    policy = VLLMPolicy(
        engine=engine,
        env=env,
        task_id=spec.task_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_history_steps=max_history_steps,
        lora_request=lora_request,
    )
    try:
        result = runner.run(
            policy,
            seed=spec.seed,
            task_id=spec.task_id,
            scenario_family=spec.scenario_family,
            max_steps=max_steps,
        )
    except Exception:  # noqa: BLE001
        LOG.exception("rollout failed: task=%s seed=%s", spec.task_id, spec.seed)
        raise
    return PPORolloutOutput(spec=spec, result=result, messages=policy.messages)


def run_rollout_batch(
    specs: list[PPORolloutSpec],
    *,
    engine: VLLMEngine,
    lora_request: Any | None,
    max_workers: int = 16,
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    max_history_steps: int = 24,
    max_steps: int | None = None,
) -> list[PPORolloutOutput]:
    """Collect a batch of rollouts in parallel.

    `lora_request` is shared across all threads — every rollout uses the
    same adapter, which is what we want during a single PPO step. The
    trainer updates this between steps by saving a new adapter and
    constructing a new `LoRARequest`.

    Returns rollouts in the SAME ORDER as `specs` (so trainers can
    correlate spec index with rollout index without an explicit key).
    """
    if not specs:
        return []

    outputs: list[PPORolloutOutput | None] = [None] * len(specs)
    workers = max(1, min(max_workers, len(specs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_one,
                spec,
                engine=engine,
                lora_request=lora_request,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_history_steps=max_history_steps,
                max_steps=max_steps,
            ): i
            for i, spec in enumerate(specs)
        }
        for future in as_completed(futures):
            i = futures[future]
            outputs[i] = future.result()

    assert all(o is not None for o in outputs)
    return outputs  # type: ignore[return-value]


__all__ = [
    "PPORolloutOutput",
    "PPORolloutSpec",
    "run_rollout_batch",
]
