from __future__ import annotations

from types import SimpleNamespace

from training.ppo.preflight_vllm import _summarize_rollout_result


def test_summarize_rollout_result_uses_last_step_done():
    result = SimpleNamespace(
        steps=2,
        success=False,
        total_reward=0.25,
        trajectory=[
            SimpleNamespace(done=False, parse_error=None),
            SimpleNamespace(done=True, parse_error="bad json"),
        ],
    )

    assert _summarize_rollout_result(result) == {
        "steps": 2,
        "success": False,
        "done": True,
        "total_reward": 0.25,
        "parse_errors": 1,
    }
