"""Parse model output into a typed RawDroneAction.

Several formats are accepted, in priority order when raw text is parsed:

1. `<tool_call>{...}</tool_call>` XML wrapper (Qwen3, Hermes, Llama-3.1
   tool-call native template).
2. JSON text (returned as a chat completion's `content`):
   `{"tool": "fly_to_viewpoint", "args": {"x": 30, ...}}`
   `{"tool_name": "fly_to_viewpoint", "arguments": {...}}` (OpenAI-style)
3. OpenAI/Anthropic tool_calls (a list of structured tool-call dicts):
   `[{"type": "function", "function": {"name": "x", "arguments": "{...}"}}]`
   `[{"type": "tool_use", "name": "x", "input": {...}}]`

Reasoning-mode preprocessing: any `<think>...</think>` blocks are stripped
before tool-call extraction. Their content is preserved on
`parse_action_with_thinking`'s return value so audit logs can record what
the model reasoned about.

The parser recovers from common LLM quirks: code-fenced JSON, leading/
trailing prose, unwrapped tool-call dicts, string-encoded `arguments`,
multiple JSON objects per response. Unrecoverable input becomes
ActionValidationError so the env loop turns it into a structured "format
invalid" turn instead of crashing.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any, NamedTuple

from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import RawDroneAction


# Capture group 1 = code-fenced body if present, else the whole match.
_CODE_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
# Reasoning-mode block (Qwen3 / DeepSeek-R1 / o1-style).
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
# Native tool-call wrapper used by Qwen3, Hermes, Llama-3.1 chat templates.
_TOOL_CALL_BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)
# Llama "ipython" tool-call format: `name(arg1=val1, arg2=val2)`. The model
# emits this when given tool schemas via Meta's native API; the args are
# literal Python values so we parse with `ast.literal_eval` for safety.
_PYTHON_CALL = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)\s*$", re.DOTALL)


class ParsedAction(NamedTuple):
    """Action plus the reasoning-block prose that preceded it (if any).

    `thinking` is the joined text of every `<think>...</think>` block stripped
    during parsing — empty string when the model didn't emit any. Audit logs
    capture this so we can review the model's reasoning without re-parsing
    the raw response.
    """

    action: RawDroneAction
    thinking: str


def parse_action(payload: Any) -> RawDroneAction:
    """Coerce model output into a RawDroneAction or raise ActionValidationError.

    Accepts:
    - RawDroneAction (passthrough)
    - dict with `tool`/`tool_name` and `args`/`arguments`
    - list of OpenAI/Anthropic tool_call dicts (uses the first)
    - str containing JSON, fenced JSON, or `<tool_call>...</tool_call>`,
      with optional `<think>...</think>` reasoning blocks anywhere
    """

    return parse_action_with_thinking(payload).action


def parse_action_with_thinking(payload: Any) -> ParsedAction:
    """Like `parse_action` but also returns any `<think>` content that was
    stripped from the input. Useful for audit logging and reasoning-mode
    SFT data. For non-string payloads `thinking` is always the empty string.
    """

    if isinstance(payload, RawDroneAction):
        return ParsedAction(payload, "")
    if isinstance(payload, list):
        return ParsedAction(_parse_tool_calls(payload), "")
    if isinstance(payload, dict):
        return ParsedAction(_parse_dict(payload), "")
    if isinstance(payload, str):
        return _parse_str(payload)
    raise ActionValidationError(
        f"unsupported action payload type: {type(payload).__name__}"
    )


# --- format dispatch ---------------------------------------------------------


def _parse_str(text: str) -> ParsedAction:
    """Pull the first tool call out of a possibly-prosey completion string.

    Pipeline (each stage tried in order; first success wins):
      1. Strip every `<think>...</think>` block; keep joined content as `thinking`.
      2. `<tool_call>{...}</tool_call>` XML wrapper (Qwen3 native format).
      3. Direct JSON parse of the cleaned text.
      4. Code-fenced JSON (` ```json ... ``` `).
      5. First balanced `{...}` object scanned with brace depth (NOT a greedy
         regex — that bug spanned `<think>` content into the tool-call JSON).
    """

    cleaned, thinking = _strip_think_blocks(text)
    cleaned = cleaned.strip()
    if not cleaned:
        raise ActionValidationError("model emitted reasoning but no tool call")

    # 1. Native <tool_call> XML wrapper.
    tool_call_match = _TOOL_CALL_BLOCK.search(cleaned)
    if tool_call_match:
        body = tool_call_match.group(1).strip()
        parsed = _try_json(body)
        if parsed is not None:
            return ParsedAction(_parse_dict_or_call(parsed), thinking)

    # 2. Llama "ipython" Python-call format: `name(k1=v1, k2=v2)`.
    py_action = _try_python_call(cleaned)
    if py_action is not None:
        return ParsedAction(py_action, thinking)

    # 3. Direct JSON parse of cleaned text.
    parsed = _try_json(cleaned)
    if parsed is not None:
        return ParsedAction(_parse_dict_or_call(parsed), thinking)

    # 4. Fenced JSON.
    fence_match = _CODE_FENCE.search(cleaned)
    if fence_match:
        parsed = _try_json(fence_match.group(1))
        if parsed is not None:
            return ParsedAction(_parse_dict_or_call(parsed), thinking)

    # 5. First balanced {...} object — brace-depth scanner, not a greedy regex.
    body = _first_json_object(cleaned)
    if body is not None:
        parsed = _try_json(body)
        if parsed is not None:
            return ParsedAction(_parse_dict_or_call(parsed), thinking)

    raise ActionValidationError(
        f"could not extract a tool call from model output: {cleaned[:160]!r}"
    )


def _try_python_call(text: str) -> RawDroneAction | None:
    """Parse `name(k1=v1, k2=v2, ...)` (Llama ipython tool format).

    Uses `ast` so we only ever evaluate literal Python values (numbers,
    strings, lists, tuples, dicts, booleans, None) — never function calls
    or attribute access. Returns None on any failure so the caller can fall
    through to the next stage.
    """

    match = _PYTHON_CALL.match(text)
    if not match:
        return None
    try:
        node = ast.parse(text.strip(), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    if node.args:
        # Llama always uses keyword arguments for tool calls; positional
        # args here are unexpected and ambiguous — fall through.
        return None
    args: dict[str, Any] = {}
    for kw in node.keywords:
        if kw.arg is None:
            return None  # **kwargs splat is not a tool call shape
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return RawDroneAction(tool_name=node.func.id, arguments=args)


def _strip_think_blocks(text: str) -> tuple[str, str]:
    """Remove `<think>...</think>` segments and return (rest, joined_thinking).

    Multiple think blocks are joined with newlines preserving order. Tag
    matching is case-insensitive (some templates emit `<Think>`).
    """

    matches = _THINK_BLOCK.findall(text)
    if not matches:
        return text, ""
    thinking = "\n".join(m.strip() for m in matches if m.strip())
    rest = _THINK_BLOCK.sub("", text)
    return rest, thinking


def _first_json_object(text: str) -> str | None:
    """Return the first complete `{...}` substring by scanning brace depth.

    Skips braces inside JSON string literals so an unbalanced brace inside
    a string doesn't confuse the depth counter. Returns None if no complete
    object is found.
    """

    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start : i + 1]
    return None


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
