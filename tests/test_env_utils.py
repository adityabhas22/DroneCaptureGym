from __future__ import annotations

from training.env_utils import load_dotenv_if_present, visible_token_names


def test_load_dotenv_key_value_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("HF_TOKEN=from_file\n", encoding="utf-8")

    assert load_dotenv_if_present(env_path) == ["HF_TOKEN"]
    assert visible_token_names() == ["HF_TOKEN"]

    monkeypatch.setenv("HF_TOKEN", "already_set")
    assert load_dotenv_if_present(env_path) == []


def test_load_dotenv_single_raw_token_as_hf_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("raw_token_value\n", encoding="utf-8")

    assert load_dotenv_if_present(env_path) == ["HF_TOKEN"]
    assert visible_token_names() == ["HF_TOKEN"]
