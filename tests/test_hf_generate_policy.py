"""CPU-stubbed tests for HFGeneratePolicy.

We verify the message-construction and prompt-rendering plumbing using a
fake model + fake tokenizer that mimic the HF interface but do not run
any real generation. A real GPU smoke is gated behind ``RUN_GPU_TESTS``
because Qwen3-4B is too big to load on a CI box.
"""

from __future__ import annotations

import os

import pytest

from dronecaptureops.agent.hf_generate_policy import HFGeneratePolicy
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


class _FakeTokenizer:
    """Bare-minimum tokenizer surface that HFGeneratePolicy uses."""

    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        # Produce a deterministic string we can grep in assertions.
        rendered = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            rendered.append(f"<{role}>{content}</{role}>")
        if add_generation_prompt:
            rendered.append("<assistant>")
        return "".join(rendered)

    def __call__(self, prompt, *, return_tensors=None, add_special_tokens=False):
        # Return a tiny mock with .input_ids and .attention_mask shaped
        # like a transformers BatchEncoding.
        import torch

        ids = torch.zeros((1, 4), dtype=torch.long)
        mask = torch.ones((1, 4), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": mask}

    def decode(self, ids, *, skip_special_tokens=True):
        return "{}"


class _FakeModel:
    training = False

    @property
    def device(self):
        import torch

        return torch.device("cpu")

    def generate(self, *args, **kwargs):
        import torch

        # Append a single token to simulate generation; total length = input + 1.
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        new = torch.zeros((1, input_ids.shape[1] + 1), dtype=torch.long)
        return new

    def eval(self):
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self


def test_message_initialisation_includes_system_and_tool_schema():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=0, task="basic_thermal_survey")
    policy = HFGeneratePolicy(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        env=env,
        task_id="basic_thermal_survey",
    )
    # Force initialisation through the protected helper since we're not
    # running a full rollout in this CPU test.
    policy._ensure_initialised()  # noqa: SLF001
    assert policy.messages, "system message should be present after initialisation"
    sys_msg = policy.messages[0]
    assert sys_msg["role"] == "system"
    assert "Tool JSON Schemas" in sys_msg["content"]


def test_render_prompt_is_apply_chat_template_compatible():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=0, task="basic_thermal_survey")
    tokenizer = _FakeTokenizer()
    policy = HFGeneratePolicy(
        model=_FakeModel(),
        tokenizer=tokenizer,
        env=env,
        task_id="basic_thermal_survey",
    )
    policy._ensure_initialised()  # noqa: SLF001
    rendered = policy._render_prompt(policy._messages)  # noqa: SLF001
    # The fake tokenizer surrounds messages with role tags and adds a
    # final <assistant> generation marker.
    assert rendered.startswith("<system>")
    assert rendered.endswith("<assistant>")


def test_prepare_prompt_then_ingest_completion_round_trip():
    """Group-batched protocol contract.

    ``prepare_prompt`` should append the user turn and snapshot the
    bounded prompt window, leaving ``_pending_prompt_messages`` set.
    ``ingest_completion`` should consume the pending window, append the
    assistant turn, and update ``training_messages`` to exactly the
    prompt+completion pair that the model would have scored if we had
    called ``next_action`` instead.
    """

    from dronecaptureops.agent.policies import AgentContext

    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=0, task="basic_thermal_survey")
    policy = HFGeneratePolicy(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        env=env,
        task_id="basic_thermal_survey",
    )
    context = AgentContext()

    prompt = policy.prepare_prompt(obs, context)
    assert isinstance(prompt, str) and prompt.endswith("<assistant>")
    assert policy._pending_prompt_messages is not None  # noqa: SLF001
    pending_len = len(policy._pending_prompt_messages)  # noqa: SLF001

    completion = (
        '{"tool_name": "get_mission_checklist", "arguments": {}}'
    )
    action = policy.ingest_completion(completion)
    assert action.tool_name == "get_mission_checklist"
    assert policy._pending_prompt_messages is None  # noqa: SLF001
    # training_messages should be the bounded prompt window plus exactly
    # one assistant turn carrying the raw completion text.
    assert len(policy.training_messages) == pending_len + 1
    assert policy.training_messages[-1] == {
        "role": "assistant",
        "content": completion,
    }


def test_double_prepare_without_ingest_raises():
    from dronecaptureops.agent.policies import AgentContext

    env = DroneCaptureOpsEnvironment()
    obs = env.reset(seed=0, task="basic_thermal_survey")
    policy = HFGeneratePolicy(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        env=env,
        task_id="basic_thermal_survey",
    )
    context = AgentContext()
    policy.prepare_prompt(obs, context)
    with pytest.raises(RuntimeError, match="prepare_prompt"):
        policy.prepare_prompt(obs, context)


def test_ingest_without_prepare_raises():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=0, task="basic_thermal_survey")
    policy = HFGeneratePolicy(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        env=env,
        task_id="basic_thermal_survey",
    )
    with pytest.raises(RuntimeError, match="prepare_prompt"):
        policy.ingest_completion("{}")


def test_trimmed_one_history_step_keeps_current_observation_only():
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=0, task="basic_thermal_survey")
    policy = HFGeneratePolicy(
        model=_FakeModel(),
        tokenizer=_FakeTokenizer(),
        env=env,
        task_id="basic_thermal_survey",
        max_history_steps=1,
    )
    policy._messages = [  # noqa: SLF001
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial observation"},
        {"role": "assistant", "content": "{}"},
        {"role": "user", "content": "current observation"},
    ]

    trimmed = policy._trimmed()  # noqa: SLF001

    assert trimmed == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "current observation"},
    ]


@pytest.mark.skipif(
    not os.environ.get("RUN_GPU_TESTS"),
    reason="Real-model smoke requires a GPU and is opt-in via RUN_GPU_TESTS=1.",
)
def test_real_model_smoke():
    """Optional: load a small HF model and run a single generation."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = os.environ.get("GRPO_TEST_MODEL", "sshleifer/tiny-gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    env = DroneCaptureOpsEnvironment()
    env.reset(seed=0, task="basic_thermal_survey")
    policy = HFGeneratePolicy(
        model=model,
        tokenizer=tokenizer,
        env=env,
        task_id="basic_thermal_survey",
        max_tokens=8,
    )
    policy._ensure_initialised()  # noqa: SLF001
    text = policy._generate("Hello there:")  # noqa: SLF001
    assert isinstance(text, str)
