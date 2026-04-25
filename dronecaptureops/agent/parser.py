"""Parse model output into a typed RawDroneAction.

Two formats are accepted:

1. JSON text (returned as a chat completion's `content`):
   `{"tool": "fly_to_viewpoint", "args": {"x": 30, ...}}`
   `{"tool_name": "fly_to_viewpoint", "arguments": {...}}` (OpenAI-style)
2. OpenAI/Anthropic tool_calls (a list of structured tool-call dicts):
   `[{"type": "function", "function": {"name": "x", "arguments": "{...}"}}]`
   `[{"type": "tool_use", "name": "x", "input": {...}}]`

The parser recovers from common LLM quirks: code-fenced JSON, leading/
trailing prose, unwrapped tool-call dicts, string-encoded `arguments`.
Unrecoverable input becomes ActionValidationError so the env loop turns it
into a structured "format invalid" turn instead of crashing.
"""

from __future__ import annotations

import json
import re
from typing import Any

from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import RawDroneAction


# Capture group 1 = code-fenced body if present, else the whole match.
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# Loose JSON object detector — picks the first {...} that parses.
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def parse_action(payload: Any) -> RawDroneAction:
    """Coerce model output into a RawDroneAction or raise ActionValidationError.

    Accepts:
    - RawDroneAction (passthrough)
    - dict with `tool`/`tool_name` and `args`/`arguments`
    - list of OpenAI/Anthropic tool_call dicts (uses the first)
    - str containing JSON or fenced JSON
    """

    if isinstance(payload, RawDroneAction):
        return payload
    if isinstance(payload, list):
        return _parse_tool_calls(payload)
    if isinstance(payload, dict):
        return _parse_dict(payload)
    if isinstance(payload, str):
        return _parse_str(payload)
    raise ActionValidationError(
        f"unsupported action payload type: {type(payload).__name__}"
    )


# --- format dispatch ---------------------------------------------------------


def _parse_str(text: str) -> RawDroneAction:
    """Pull the first JSON object out of a possibly-prosey completion string."""

    cleaned = text.strip()
    if not cleaned:
        raise ActionValidationError("empty model output")

    # Try direct parse first.
    parsed = _try_json(cleaned)
    if parsed is not None:
        return _parse_dict_or_call(parsed)

    # Look for fenced JSON.
    fence_match = _CODE_FENCE.search(cleaned)
    if fence_match:
        parsed = _try_json(fence_match.group(1))
        if parsed is not None:
            return _parse_dict_or_call(parsed)

    # Last resort: find the first balanced {...} substring.
    object_match = _JSON_OBJECT.search(cleaned)
    if object_match:
        parsed = _try_json(object_match.group(0))
        if parsed is not None:
            return _parse_dict_or_call(parsed)

    raise ActionValidationError(
        f"could not extract a JSON tool call from model output: {cleaned[:120]!r}"
    )


def _parse_dict_or_call(value: Any) -> RawDroneAction:
    if isinstance(value, list):
        return _parse_tool_calls(value)
    if isinstance(value, dict):
        # Single tool-call object emitted as a dict (some models do this).
        if _looks_like_tool_call(value):
            return _from_tool_call(value)
        return _parse_dict(value)
    raise ActionValidationError(f"unexpected JSON payload: {type(value).__name__}")


def _parse_dict(value: dict[str, Any]) -> RawDroneAction:
    """Accept either {tool, args} or {tool_name, arguments}."""

    name = value.get("tool_name") or value.get("tool") or value.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ActionValidationError(f"tool name missing or invalid in payload: {_summarise(value)}")
    raw_args = value.get("arguments")
    if raw_args is None:
        raw_args = value.get("args")
    if raw_args is None:
        raw_args = value.get("input")
    if raw_args is None:
        raw_args = {}
    args = _coerce_args(raw_args, tool_name=name.strip())
    return RawDroneAction(tool_name=name.strip(), arguments=args)


def _parse_tool_calls(calls: list[Any]) -> RawDroneAction:
    """Take the first tool call from an OpenAI/Anthropic-style list."""

    if not calls:
        raise ActionValidationError("empty tool_calls list")
    first = calls[0]
    if not isinstance(first, dict):
        raise ActionValidationError(
            f"tool_calls[0] is not a dict: {type(first).__name__}"
        )
    return _from_tool_call(first)


def _from_tool_call(call: dict[str, Any]) -> RawDroneAction:
    """Extract a tool call from an OpenAI or Anthropic structured dict."""

    function = call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        args = function.get("arguments")
    else:
        # Anthropic-style: {"type": "tool_use", "name": ..., "input": {...}}
        name = call.get("name")
        args = call.get("input")
        if args is None:
            args = call.get("arguments")
    if not isinstance(name, str) or not name.strip():
        raise ActionValidationError(f"tool_call missing function.name: {_summarise(call)}")
    args = _coerce_args(args, tool_name=name.strip())
    return RawDroneAction(tool_name=name.strip(), arguments=args)


# --- helpers -----------------------------------------------------------------


def _coerce_args(raw: Any, *, tool_name: str) -> dict[str, Any]:
    """Normalise tool arguments to a dict, accepting JSON-string encodings."""

    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ActionValidationError(
                f"{tool_name}: arguments string is not valid JSON: {raw[:120]!r}"
            ) from exc
        if not isinstance(decoded, dict):
            raise ActionValidationError(
                f"{tool_name}: arguments must be a JSON object, got {type(decoded).__name__}"
            )
        return decoded
    raise ActionValidationError(
        f"{tool_name}: arguments must be a dict or JSON string, got {type(raw).__name__}"
    )


def _looks_like_tool_call(value: dict[str, Any]) -> bool:
    """Distinguish OpenAI-style tool_call dicts from {tool, args} payloads."""

    if value.get("type") in {"function", "tool_use"}:
        return True
    if "function" in value and isinstance(value.get("function"), dict):
        return True
    return False


def _try_json(text: str) -> Any | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _summarise(value: Any) -> str:
    rendered = repr(value)
    if len(rendered) > 160:
        return rendered[:157] + "..."
    return rendered
