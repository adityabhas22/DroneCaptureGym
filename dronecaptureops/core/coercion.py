"""Argument coercion helpers for tool handlers.

Tool handlers receive `arguments: dict[str, Any]` from the agent and need to
turn untrusted JSON-ish values into typed Python primitives. A bad input
(e.g. `"altitude_m": "high"`) must surface as an `ActionValidationError`
rather than an uncaught `ValueError` that crashes the environment loop.
"""

from __future__ import annotations

from typing import Any

from dronecaptureops.core.errors import ActionValidationError


_MISSING = object()


def coerce_float(args: dict[str, Any], key: str, *, default: float | object = _MISSING, minimum: float | None = None, maximum: float | None = None) -> float:
    """Return `args[key]` as float, raising ActionValidationError on bad input."""

    if key not in args:
        if default is _MISSING:
            raise ActionValidationError(f"missing required argument: {key}")
        return float(default)  # type: ignore[arg-type]
    raw = args[key]
    if isinstance(raw, bool) or raw is None:
        raise ActionValidationError(f"invalid {key}: expected number, got {raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ActionValidationError(f"invalid {key}: cannot interpret {raw!r} as a number") from exc
    if value != value:  # NaN check
        raise ActionValidationError(f"invalid {key}: NaN is not allowed")
    if minimum is not None and value < minimum:
        raise ActionValidationError(f"{key} below minimum {minimum}: {value}")
    if maximum is not None and value > maximum:
        raise ActionValidationError(f"{key} above maximum {maximum}: {value}")
    return value


def coerce_optional_float(args: dict[str, Any], key: str) -> float | None:
    """Return `args[key]` as float or None if absent or null."""

    if key not in args or args[key] is None:
        return None
    return coerce_float(args, key)


def coerce_str(args: dict[str, Any], key: str, *, default: str | object = _MISSING, allowed: set[str] | None = None) -> str:
    """Return `args[key]` as a non-empty string with optional allowed-set check."""

    if key not in args:
        if default is _MISSING:
            raise ActionValidationError(f"missing required argument: {key}")
        return str(default)  # type: ignore[arg-type]
    raw = args[key]
    if not isinstance(raw, str) or not raw.strip():
        raise ActionValidationError(f"invalid {key}: expected non-empty string, got {raw!r}")
    value = raw.strip()
    if allowed is not None and value not in allowed:
        raise ActionValidationError(f"{key} must be one of {sorted(allowed)}, got {value!r}")
    return value


def coerce_str_list(args: dict[str, Any], key: str) -> list[str]:
    """Return `args[key]` as a list of strings (empty list if absent)."""

    if key not in args or args[key] is None:
        return []
    raw = args[key]
    if not isinstance(raw, list):
        raise ActionValidationError(f"invalid {key}: expected list of strings, got {type(raw).__name__}")
    out: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise ActionValidationError(f"invalid {key}[{index}]: expected string, got {type(item).__name__}")
        out.append(item)
    return out
