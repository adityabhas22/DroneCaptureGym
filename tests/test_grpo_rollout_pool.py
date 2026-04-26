"""CPU-stubbed test for the GRPO group-batched rollout pool.

We assert two contracts that the trainer relies on:

1. Each rollout *turn* is driven by exactly ONE ``model.generate()`` call
   that batches all active prompts. Prior implementation (per-rollout
   ``RolloutRunner.run``) made ``G * T`` separate calls per group; the
   speedup comes from collapsing those into ``T`` batched calls.
2. Output ordering is preserved: ``run_rollout_batch`` returns one entry
   per input spec in the same order, so the trainer's group-normalization
   indexing keeps working.

We use a fake tokenizer + fake model so the test runs on CPU in <1s.
The fake model's ``generate()`` records every call so we can assert the
exact batching behavior.
"""

from __future__ import annotations

import json
import random
from typing import Any

import torch

from training.grpo.rollout_pool import (
    GRPORolloutSpec,
    run_rollout_batch,
    sample_specs,
)


VALID_TOOL_CALL = (
    '{"tool_name": "get_mission_checklist", "arguments": {}}'
)


class _FakeTokenizer:
    """Bare-minimum tokenizer surface used by both prepare_prompt and
    the batched ``_batched_generate`` helper."""

    pad_token_id = 0
    eos_token_id = 1
    padding_side = "left"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        rendered = []
        for m in messages:
            content = m.get("content", "")
            rendered.append(f"<{m.get('role')}>{content}</{m.get('role')}>")
        if add_generation_prompt:
            rendered.append("<assistant>")
        return "".join(rendered)

    def __call__(
        self,
        prompts,
        *,
        return_tensors=None,
        padding=False,
        add_special_tokens=False,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]
        # Use prompt length as token length so different prompts produce
        # different shapes pre-padding.
        lengths = [max(1, len(p)) for p in prompts]
        max_len = max(lengths)
        ids = torch.zeros((len(prompts), max_len), dtype=torch.long)
        mask = torch.zeros((len(prompts), max_len), dtype=torch.long)
        for i, ln in enumerate(lengths):
            mask[i, max_len - ln :] = 1  # left-padded
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, *, skip_special_tokens=True):
        # Every generated row decodes to a valid get_mission_checklist call.
        return VALID_TOOL_CALL


class _RecordingModel:
    """Fake model whose ``generate`` records every call's batch size."""

    training = False
    generate_calls: list[int]

    def __init__(self) -> None:
        self.generate_calls = []

    @property
    def device(self):
        return torch.device("cpu")

    def generate(self, *, input_ids, attention_mask=None, **kwargs):
        self.generate_calls.append(int(input_ids.shape[0]))
        # Append max_new_tokens of zeros — _batched_generate slices them
        # off the prompt so the decoded text is fully controlled by the
        # tokenizer's stub ``decode``.
        max_new = int(kwargs.get("max_new_tokens", 8))
        suffix = torch.zeros(
            (input_ids.shape[0], max_new), dtype=input_ids.dtype
        )
        return torch.cat([input_ids, suffix], dim=1)

    def eval(self):
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self


def test_run_rollout_batch_batches_one_call_per_turn():
    """For one group of size G, every turn should issue exactly one
    batched generate of width G (plus whatever turns happen until envs
    finish), not G separate calls."""

    rng = random.Random(0)
    specs = sample_specs(
        rng=rng,
        train_tasks=["basic_thermal_survey"],
        prompts_per_step=1,
        group_size=3,
    )
    assert len(specs) == 3
    model = _RecordingModel()
    tokenizer = _FakeTokenizer()

    outputs = run_rollout_batch(
        specs,
        model=model,
        tokenizer=tokenizer,
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        max_history_steps=1,
        max_steps=2,
    )

    assert len(outputs) == 3
    assert all(o is not None for o in outputs)
    # Every generate call must batch all active rollouts; with no env
    # ever marking done early the call width is exactly G on every turn.
    assert model.generate_calls, "no generate calls were recorded"
    assert all(
        n == 3 for n in model.generate_calls
    ), f"expected width-3 batches, got {model.generate_calls}"


def test_run_rollout_batch_preserves_spec_order_across_groups():
    """Two prompts × group=2 should return outputs in spec-major order
    so the trainer's prompt_index group-normalization stays correct."""

    rng = random.Random(0)
    specs = sample_specs(
        rng=rng,
        train_tasks=["basic_thermal_survey"],
        prompts_per_step=2,
        group_size=2,
    )
    assert len(specs) == 4
    model = _RecordingModel()
    tokenizer = _FakeTokenizer()

    outputs = run_rollout_batch(
        specs,
        model=model,
        tokenizer=tokenizer,
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        max_history_steps=1,
        max_steps=1,
    )

    assert len(outputs) == 4
    assert all(o is not None for o in outputs)
    # Ordering: outputs[i].spec must equal specs[i].
    for i, output in enumerate(outputs):
        assert output is not None
        assert output.spec.prompt_index == specs[i].prompt_index
        assert output.spec.group_index == specs[i].group_index
        assert output.spec.seed == specs[i].seed


def test_run_rollout_batch_messages_match_training_window():
    """Each output's messages should be the bounded prompt+completion
    window used for training (not the raw full history). The last
    message must be the assistant's completion text."""

    rng = random.Random(0)
    specs = sample_specs(
        rng=rng,
        train_tasks=["basic_thermal_survey"],
        prompts_per_step=1,
        group_size=2,
    )
    model = _RecordingModel()
    tokenizer = _FakeTokenizer()

    outputs = run_rollout_batch(
        specs,
        model=model,
        tokenizer=tokenizer,
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        max_history_steps=1,
        max_steps=1,
    )

    for output in outputs:
        assert output is not None
        assert output.messages, "training_messages should not be empty"
        last = output.messages[-1]
        assert last["role"] == "assistant"
        # The fake tokenizer always decodes to VALID_TOOL_CALL which is
        # what HFGeneratePolicy.ingest_completion records on the assistant
        # turn.
        assert last["content"] == VALID_TOOL_CALL
        # And the action JSON should round-trip into a tool name we
        # recognize in the trajectory.
        assert output.result.trajectory
        assert (
            output.result.trajectory[0].action.get("tool_name")
            == "get_mission_checklist"
        ), output.result.trajectory[0].action


def test_run_rollout_batch_empty_specs():
    model = _RecordingModel()
    outputs = run_rollout_batch(
        [],
        model=model,
        tokenizer=_FakeTokenizer(),
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        max_history_steps=1,
        max_steps=1,
    )
    assert outputs == []
    assert model.generate_calls == []


def test_run_rollout_batch_single_group_single_spec():
    """Smoke that a 1-prompt 1-sample group still works (G=1 batch)."""

    spec = GRPORolloutSpec(
        task_id="basic_thermal_survey",
        seed=7,
        prompt_index=0,
        group_index=0,
    )
    model = _RecordingModel()
    outputs = run_rollout_batch(
        [spec],
        model=model,
        tokenizer=_FakeTokenizer(),
        temperature=0.0,
        top_p=1.0,
        max_tokens=8,
        max_history_steps=1,
        max_steps=1,
    )
    assert len(outputs) == 1 and outputs[0] is not None
    assert model.generate_calls == [1]
    # The action JSON we control in the fake tokenizer must round-trip.
    payload = json.loads(VALID_TOOL_CALL)
    assert outputs[0].result.trajectory[0].action["tool_name"] == payload["tool_name"]
