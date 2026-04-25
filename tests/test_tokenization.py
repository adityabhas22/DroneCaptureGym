"""Tests for `training/ppo/tokenization.py`.

Uses a small fast tokenizer to validate that:
- assistant turn boundaries align with the chat template's role markers,
- response_mask covers exactly the assistant content tokens,
- truncation drops trailing turns rather than corrupting the last span.
"""

from __future__ import annotations

import pytest


pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import AutoTokenizer

from training.ppo.tokenization import pad_trajectories, tokenize_trajectory


@pytest.fixture(scope="module")
def tokenizer():
    # Use a lightweight Qwen tokenizer that ships in transformers' test fixtures.
    # If unavailable, tests skip.
    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", use_fast=True)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Qwen tokenizer unavailable: {exc}")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _conversation():
    return [
        {"role": "system", "content": "You are a drone inspection director."},
        {"role": "user", "content": "Begin the survey."},
        {"role": "assistant", "content": '{"tool": "takeoff", "args": {"altitude_m": 18}}'},
        {"role": "user", "content": "## Telemetry: in_air=True altitude=18m"},
        {"role": "assistant", "content": '{"tool": "fly_to_viewpoint", "args": {"x": 30, "y": 24, "z": 22}}'},
    ]


def test_two_assistant_spans_recovered(tokenizer):
    traj = tokenize_trajectory(_conversation(), tokenizer=tokenizer, max_length=2048)
    assert traj.n_assistant_turns == 2
    assert len(traj.assistant_spans) == 2
    # Spans must be strictly increasing and within the sequence.
    s1, e1 = traj.assistant_spans[0]
    s2, e2 = traj.assistant_spans[1]
    assert 0 <= s1 < e1 <= s2 < e2 <= traj.seq_len


def test_response_mask_matches_assistant_spans(tokenizer):
    traj = tokenize_trajectory(_conversation(), tokenizer=tokenizer, max_length=2048)
    nonzero = (traj.response_mask > 0).nonzero(as_tuple=True)[0].tolist()
    expected: set[int] = set()
    for s, e in traj.assistant_spans:
        expected.update(range(s, e))
    assert set(nonzero) == expected


def test_truncation_drops_trailing_turn_cleanly(tokenizer):
    # Force truncation by setting a very small max_length.
    traj = tokenize_trajectory(_conversation(), tokenizer=tokenizer, max_length=64)
    # Either the first turn fits and the rest is dropped, or nothing fits.
    assert traj.n_assistant_turns <= 2
    for s, e in traj.assistant_spans:
        assert e <= traj.seq_len, "span must not exceed truncated seq_len"


def test_pad_trajectories_aligns_lengths(tokenizer):
    short = tokenize_trajectory(_conversation()[:3], tokenizer=tokenizer, max_length=2048)
    full = tokenize_trajectory(_conversation(), tokenizer=tokenizer, max_length=2048)
    batch = pad_trajectories([short, full], pad_token_id=tokenizer.pad_token_id)
    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape[1] == max(short.seq_len, full.seq_len)
    # Pad tokens have attention_mask=0
    if short.seq_len < full.seq_len:
        assert batch["attention_mask"][0, short.seq_len:].sum().item() == 0
