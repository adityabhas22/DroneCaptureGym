"""Small environment helpers for local training/eval CLIs.

We intentionally avoid adding a python-dotenv dependency. The loader only
handles simple KEY=VALUE lines, never overwrites existing environment
variables, and callers should only log whether secrets are present.
"""

from __future__ import annotations

import os
from pathlib import Path


TOKEN_ENV_NAMES = ("HF_TOKEN", "HF_AUTH_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def load_dotenv_if_present(path: Path) -> list[str]:
    """Load simple KEY=VALUE pairs from `path` without overriding env vars."""

    if not path.exists():
        return []

    loaded: list[str] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    meaningful_lines = [line for line in lines if line and not line.startswith("#")]
    if len(meaningful_lines) == 1 and "=" not in meaningful_lines[0]:
        if "HF_TOKEN" not in os.environ:
            os.environ["HF_TOKEN"] = meaningful_lines[0].strip().strip('"').strip("'")
            loaded.append("HF_TOKEN")
        return loaded

    for line in lines:
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value
        loaded.append(key)
    return loaded


def visible_token_names() -> list[str]:
    """Return configured token variable names only; never expose values."""

    return [name for name in TOKEN_ENV_NAMES if os.environ.get(name)]


def hf_token_visible() -> bool:
    return bool(visible_token_names())


__all__ = ["TOKEN_ENV_NAMES", "hf_token_visible", "load_dotenv_if_present", "visible_token_names"]
