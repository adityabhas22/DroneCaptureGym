"""Seed helpers."""

from __future__ import annotations

from dronecaptureops.core.constants import DEFAULT_SEED


def normalize_seed(seed: int | None) -> int:
    """Return the default seed when no seed is supplied."""

    return DEFAULT_SEED if seed is None else int(seed)
