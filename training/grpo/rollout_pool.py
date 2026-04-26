"""Sequential rollout collector for GRPO training.

Unlike the PPO branch we deliberately do NOT use a thread pool here:

- The whole point of the GRPO branch is to escape vLLM's V1 thread-
  unsafe ``LLM.generate()`` and the L40S OOM caused by colocating vLLM
  with the training model. Generation now runs through a shared
  ``model.generate()`` on the same in-process model.
- HF transformers ``generate()`` is not safe to call concurrently from
  multiple threads either (it mutates the model's KV cache state).
- Total rollouts per step are tiny anyway (e.g. 4 prompts × 4 group
  size = 16), and each rollout already contains ``max_episode_steps``
  serial generation calls, so any cross-rollout parallelism would
  contend on the same GPU.

So the pool runs episodes one at a time. Order is preserved; the
trainer relies on the layout ``[p0_g0, p0_g1, ..., p0_gG-1, p1_g0, ...]``
to compute group-normalized advantages.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

from dronecaptureops.agent.hf_generate_policy import HFGeneratePolicy
from dronecaptureops.agent.rollout import RolloutResult, RolloutRunner
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


LOG = logging.getLogger("dronecaptureops.grpo.rollout_pool")


@dataclass
class GRPORolloutSpec:
    """One rollout request — task + seed + group bookkeeping."""

    task_id: str | None
    seed: int
    prompt_index: int            # which prompt within the step (0..N-1)
    group_index: int             # which sample within the group (0..G-1)
    scenario_family: str | None = None


@dataclass
class GRPORolloutOutput:
    """Everything the trainer needs from one completed rollout."""

    spec: GRPORolloutSpec
    result: RolloutResult
    messages: list[dict[str, Any]]


def sample_specs(
    *,
    rng: random.Random,
    train_tasks: list[str],
    prompts_per_step: int,
    group_size: int,
) -> list[GRPORolloutSpec]:
    """Sample G rollouts for each of N prompts, laid out group-major.

    Each prompt picks a task; all G samples within that prompt use the
    same task but different seeds, so the group of rewards reflects
    sampling noise + policy stochasticity rather than task variance.
    """

    if not train_tasks:
        raise ValueError("no training tasks available for sampling")
    specs: list[GRPORolloutSpec] = []
    for prompt_idx in range(prompts_per_step):
        task_id = rng.choice(train_tasks)
        for group_idx in range(group_size):
            specs.append(
                GRPORolloutSpec(
                    task_id=task_id,
                    seed=rng.randint(0, 2**31 - 1),
                    prompt_index=prompt_idx,
                    group_index=group_idx,
                )
            )
    return specs


def run_rollout_batch(
    specs: list[GRPORolloutSpec],
    *,
    model: Any,
    tokenizer: Any,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_history_steps: int,
    max_steps: int,
    fail_fast: bool = False,
) -> list[GRPORolloutOutput | None]:
    """Run rollouts sequentially. Returns one entry per spec (None on failure).

    A ``None`` entry preserves the spec ordering so the trainer can decide
    whether to drop the failed rollout from its group or fail the step.
    """

    if not specs:
        return []

    outputs: list[GRPORolloutOutput | None] = []
    for spec in specs:
        try:
            outputs.append(
                _run_one(
                    spec,
                    model=model,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    max_history_steps=max_history_steps,
                    max_steps=max_steps,
                )
            )
        except Exception:  # noqa: BLE001
            LOG.exception(
                "rollout failed: task=%s seed=%s prompt=%d group=%d",
                spec.task_id,
                spec.seed,
                spec.prompt_index,
                spec.group_index,
            )
            if fail_fast:
                raise
            outputs.append(None)
    return outputs


def _run_one(
    spec: GRPORolloutSpec,
    *,
    model: Any,
    tokenizer: Any,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_history_steps: int,
    max_steps: int,
) -> GRPORolloutOutput:
    env = DroneCaptureOpsEnvironment()
    runner = RolloutRunner(env=env)
    policy = HFGeneratePolicy(
        model=model,
        tokenizer=tokenizer,
        env=env,
        task_id=spec.task_id,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        max_history_steps=max_history_steps,
    )
    result = runner.run(
        policy,
        seed=spec.seed,
        task_id=spec.task_id,
        scenario_family=spec.scenario_family,
        max_steps=max_steps,
    )
    return GRPORolloutOutput(spec=spec, result=result, messages=policy.messages)


__all__ = [
    "GRPORolloutOutput",
    "GRPORolloutSpec",
    "run_rollout_batch",
    "sample_specs",
]
