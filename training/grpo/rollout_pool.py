"""Group-batched rollout collector for GRPO training.

Why group-batched, not per-rollout sequential:

- The G rollouts within a single GRPO group share the same task and only
  differ by sampling seed. They can be advanced in lockstep so every
  generation turn batches G prompts through ONE ``model.generate()``
  call instead of G sequential calls.
- That cuts ``model.generate()`` invocations per step from
  ``prompts_per_step * group_size * max_episode_steps`` to
  ``prompts_per_step * max_episode_steps`` (4-8x fewer calls in practice).
- HF transformers ``generate()`` is not safe to call concurrently from
  multiple threads (it mutates the model's KV cache state) so we keep
  the calls serial within a step — the speedup comes from batching, not
  threading.

Group ordering is preserved: ``run_rollout_batch`` returns outputs in the
SAME order as the input specs so the trainer's group-normalization
indexing (``prompt_index``) keeps working unchanged.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any

from dronecaptureops.agent.hf_generate_policy import HFGeneratePolicy
from dronecaptureops.agent.policies import AgentContext
from dronecaptureops.agent.rollout import RolloutResult, RolloutStep, _reward_delta
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.errors import ActionValidationError


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
    """Run all groups, batched within each group, returning one entry per spec.

    ``specs`` is expected to be group-major (G consecutive specs per
    prompt_index). Each group is driven in lockstep through a single
    batched ``generate()`` per turn. A failed group leaves ``None`` for
    every spec in that group so the trainer can decide whether to drop
    or fail the step.
    """

    if not specs:
        return []

    # Walk the input specs and split into contiguous groups by
    # prompt_index. We keep the original index for each spec so we can
    # write outputs back in spec order regardless of how groups end up
    # being chunked.
    groups: list[list[tuple[int, GRPORolloutSpec]]] = []
    current: list[tuple[int, GRPORolloutSpec]] = []
    last_prompt: int | None = None
    for i, spec in enumerate(specs):
        if last_prompt is None or spec.prompt_index != last_prompt:
            if current:
                groups.append(current)
            current = []
            last_prompt = spec.prompt_index
        current.append((i, spec))
    if current:
        groups.append(current)

    outputs: list[GRPORolloutOutput | None] = [None] * len(specs)
    for group_pairs in groups:
        group_specs = [s for _, s in group_pairs]
        prompt_idx = group_specs[0].prompt_index if group_specs else -1
        try:
            group_outputs = _run_group(
                group_specs,
                model=model,
                tokenizer=tokenizer,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                max_history_steps=max_history_steps,
                max_steps=max_steps,
            )
        except Exception:  # noqa: BLE001
            LOG.exception(
                "group rollout failed: prompt_index=%d size=%d",
                prompt_idx,
                len(group_specs),
            )
            if fail_fast:
                raise
            continue
        for (orig_idx, _), output in zip(group_pairs, group_outputs, strict=True):
            outputs[orig_idx] = output
    return outputs


def _run_group(
    group_specs: list[GRPORolloutSpec],
    *,
    model: Any,
    tokenizer: Any,
    temperature: float,
    top_p: float,
    max_tokens: int,
    max_history_steps: int,
    max_steps: int,
) -> list[GRPORolloutOutput]:
    """Drive G envs in lockstep, batching G prompts per generation call."""

    if not group_specs:
        return []

    G = len(group_specs)
    envs = [DroneCaptureOpsEnvironment() for _ in range(G)]
    policies = [
        HFGeneratePolicy(
            model=model,
            tokenizer=tokenizer,
            env=envs[i],
            task_id=group_specs[i].task_id,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_history_steps=max_history_steps,
        )
        for i in range(G)
    ]
    contexts = [AgentContext() for _ in range(G)]
    observations = []
    initial_observations: list[dict[str, Any]] = []
    previous_breakdowns: list[dict[str, Any]] = []
    for i, spec in enumerate(group_specs):
        obs = envs[i].reset(
            seed=spec.seed,
            task=spec.task_id,
            scenario_family=spec.scenario_family,
        )
        observations.append(obs)
        initial_observations.append(obs.model_dump(mode="json"))
        previous_breakdowns.append(obs.reward_breakdown.model_dump(mode="json"))

    trajectories: list[list[RolloutStep]] = [[] for _ in range(G)]
    done_flags = [False] * G

    # Per-spec hard step cap (None falls back to env's remaining_steps).
    limits: list[int] = []
    for i in range(G):
        if max_steps:
            limits.append(int(max_steps))
        else:
            remaining = observations[i].state_summary.get("remaining_steps")
            limits.append(int(remaining) if remaining is not None else 40)
    horizon = max(limits) if limits else 0

    for turn in range(1, horizon + 1):
        active: list[int] = [
            i for i in range(G) if not done_flags[i] and turn <= limits[i]
        ]
        if not active:
            break

        prompts: list[str] = [
            policies[i].prepare_prompt(observations[i], contexts[i]) for i in active
        ]
        completions = _batched_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
        )

        for idx_in_active, i in enumerate(active):
            completion = completions[idx_in_active]
            try:
                action = policies[i].ingest_completion(completion)
            except ActionValidationError as exc:
                next_observation = envs[i].step(
                    {"tool_name": "invalid", "arguments": {}}
                )
                action_payload: dict[str, Any] = {
                    "tool_name": "invalid",
                    "arguments": {},
                }
                parse_error: str | None = str(exc)
                action = None
            else:
                next_observation = envs[i].step(action)
                action_payload = action.model_dump(mode="json")
                parse_error = None

            breakdown = next_observation.reward_breakdown.model_dump(mode="json")
            trajectories[i].append(
                RolloutStep(
                    step=turn,
                    observation=observations[i].model_dump(mode="json"),
                    action=action_payload,
                    next_observation=next_observation.model_dump(mode="json"),
                    reward=float(next_observation.reward or 0.0),
                    reward_breakdown=breakdown,
                    reward_delta=_reward_delta(previous_breakdowns[i], breakdown),
                    done=next_observation.done,
                    action_result=next_observation.action_result,
                    warnings=list(next_observation.warnings),
                    parse_error=parse_error,
                )
            )
            if action is not None:
                contexts[i].append(
                    action=action,
                    observation=next_observation,
                    action_result=next_observation.action_result,
                )
            previous_breakdowns[i] = breakdown
            observations[i] = next_observation
            if next_observation.done:
                done_flags[i] = True

    outputs: list[GRPORolloutOutput] = []
    for i, spec in enumerate(group_specs):
        final_payload = observations[i].model_dump(mode="json")
        result = RolloutResult(
            policy_name=getattr(policies[i], "name", "hf_generate"),
            seed=spec.seed,
            task_id=spec.task_id
            or final_payload.get("metadata", {}).get("task_id"),
            scenario_family=spec.scenario_family
            or final_payload.get("metadata", {}).get("scenario_family"),
            episode_id=final_payload.get("metadata", {}).get("episode_id"),
            success=bool(
                observations[i].done and observations[i].checklist_status.complete
            ),
            steps=len(trajectories[i]),
            total_reward=float(observations[i].reward or 0.0),
            reward_breakdown=observations[i].reward_breakdown.model_dump(mode="json"),
            trajectory=trajectories[i],
            initial_observation=initial_observations[i],
            final_observation=final_payload,
        )
        outputs.append(
            GRPORolloutOutput(
                spec=spec,
                result=result,
                messages=policies[i].training_messages,
            )
        )
    return outputs


def _batched_generate(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> list[str]:
    """Single batched ``model.generate()`` call across G prompts.

    Requires ``tokenizer.padding_side == 'left'`` for batched decoder
    generation to align prompt tokens to the right of the input. The
    GRPO trainer enforces this when it loads the tokenizer.
    """

    import torch

    if not prompts:
        return []

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        temperature=max(temperature, 1e-5),
        top_p=top_p,
        pad_token_id=pad_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        attention_mask=attention_mask,
    )

    prev_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model.generate(input_ids=input_ids, **gen_kwargs)
    finally:
        model.train(prev_training)

    prompt_len = input_ids.shape[1]
    new_tokens = output[:, prompt_len:]
    decoded: list[str] = []
    for i in range(new_tokens.shape[0]):
        text = tokenizer.decode(new_tokens[i], skip_special_tokens=True)
        for stop in ("<|im_end|>", "<|endoftext|>"):
            if stop in text:
                text = text.split(stop, 1)[0]
        decoded.append(text)
    return decoded


__all__ = [
    "GRPORolloutOutput",
    "GRPORolloutSpec",
    "run_rollout_batch",
    "sample_specs",
]
