"""Drop-in client for Meta's Llama API that mimics openai.OpenAI's surface.

Meta's `https://api.llama.com/v1/chat/completions` returns a different shape
than OpenAI:
- `completion_message.content` is `{"type": "text", "text": "..."}` instead
  of a flat string.
- `completion_message.stop_reason` instead of `choices[0].finish_reason`.
- `metrics: [{metric, value, ...}]` instead of `usage: {...}`.

Rather than fork `HFInferencePolicy`, we expose a wrapper with the same
`.chat.completions.create(**kwargs)` API and translate the response back to
the OpenAI shape (`SimpleNamespace`s the policy already reads). This lets
the policy and parser stay completely untouched.

Tool-call mode: Meta returns
    "tool_calls": [{"id": "...", "function": {"name": "...", "arguments": "..."}}]
which is missing the OpenAI `"type": "function"` wrapper — we add it back
so `parse_action` accepts it unchanged.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


class _APIError(Exception):
    """Raised on non-2xx responses; mimics openai.BadRequestError enough that
    `HFInferencePolicy._is_retryable` and `_serialise_api_error` can triage it.

    Handles both shapes the platform actually returns:
    - OpenAI-style:  `{"error": {"message": "...", "code": "..."}}`
    - Meta-style:    `{"title": "...", "detail": "...", "status": ...}`
    """

    def __init__(self, status_code: int, body: Any) -> None:
        message: str | None = None
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict):
                message = err.get("message")
            # Meta's native shape: top-level title/detail.
            if not message:
                detail = body.get("detail")
                title = body.get("title")
                if detail or title:
                    message = " — ".join(s for s in (title, detail) if s)
        elif isinstance(body, str):
            message = body
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code
        self.body = body
        self.response = SimpleNamespace(status_code=status_code)


class MetaLlamaClient:
    """OpenAI-shaped wrapper around `https://api.llama.com/v1/chat/completions`."""

    def __init__(self, *, api_key: str, base_url: str = "https://api.llama.com/v1", timeout: float = 60.0) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        # Strip OpenAI-only knobs Meta doesn't accept silently — most overlap.
        body = {k: v for k, v in kwargs.items() if v is not None}
        body.pop("extra_body", None)

        req = Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=self._timeout) as resp:
                raw = json.loads(resp.read())
        except HTTPError as e:
            try:
                err_body = json.loads(e.read())
            except Exception:  # noqa: BLE001
                err_body = {"error": {"message": str(e)}}
            raise _APIError(e.code, err_body) from e

        return _to_openai_shape(raw)


def _to_openai_shape(meta_response: dict[str, Any]) -> SimpleNamespace:
    """Translate Meta's completion shape to the OpenAI shape `HFInferencePolicy` reads."""

    cm = meta_response.get("completion_message") or {}

    # Content: Meta wraps as {type: text, text: ...}; flatten to a plain string.
    raw_content = cm.get("content")
    if isinstance(raw_content, dict):
        content = raw_content.get("text", "") or ""
    elif isinstance(raw_content, str):
        content = raw_content
    else:
        content = ""

    # Tool calls: Meta omits the OpenAI `"type": "function"` wrapper.
    tool_calls = cm.get("tool_calls")
    if tool_calls:
        wrapped = []
        for tc in tool_calls:
            fn = tc.get("function") or {}
            wrapped.append(
                SimpleNamespace(
                    id=tc.get("id"),
                    type=tc.get("type") or "function",
                    function=SimpleNamespace(
                        name=fn.get("name"),
                        arguments=fn.get("arguments") or "{}",
                    ),
                )
            )
        tool_calls = wrapped

    message = SimpleNamespace(role=cm.get("role") or "assistant", content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=cm.get("stop_reason"))

    metrics = {m.get("metric"): m.get("value") for m in (meta_response.get("metrics") or [])}
    usage = SimpleNamespace(
        prompt_tokens=metrics.get("num_prompt_tokens"),
        completion_tokens=metrics.get("num_completion_tokens"),
        total_tokens=metrics.get("num_total_tokens"),
    )

    response = SimpleNamespace(
        id=meta_response.get("id"),
        choices=[choice],
        usage=usage,
        model=meta_response.get("model"),
    )
    # Make `_serialise_response` happy via the model_dump fallback path.
    response.model_dump = lambda: meta_response  # type: ignore[attr-defined]
    return response


__all__ = ["MetaLlamaClient"]
