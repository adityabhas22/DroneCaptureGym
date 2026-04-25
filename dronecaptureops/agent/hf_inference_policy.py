"""HuggingFace hosted-inference policy via the OpenAI-compatible router.

Why this exists: we don't want to pay for an H200 just to evaluate base
models that are already hosted. HF's router (router.huggingface.co/v1)
exposes most chat-tuned models through providers like Together,
Fireworks, Cerebras, Replicate, etc., behind a single OpenAI-compatible
endpoint. With the user's HF_TOKEN we can drive the same eval matrix
without managing GPUs.

Robustness — this is the user-facing API for the eval, so we treat
operational failures honestly (no silent skips):
- 429 rate-limit: exponential backoff, capped retry budget
- 503 "model is currently loading": wait + retry
- transient network errors: retry
- malformed model output: surface as ActionValidationError so the
  rollout runner records a parse error (visible in diagnostics)

Auditability — every turn logs the raw response text the model
emitted, so a downstream reader can verify nothing was filtered or
post-processed before parsing. Token usage is captured per turn so
cost is transparent.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dronecaptureops.agent.llm_policies import _LLMPolicyBase, _message_to_dict
from dronecaptureops.agent.parser import parse_action_with_thinking
from dronecaptureops.agent.policies import AgentContext
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import DroneObservation, RawDroneAction


if TYPE_CHECKING:  # pragma: no cover
    pass


LOG = logging.getLogger("dronecaptureops.agent.hf")


HF_DEFAULT_BASE_URL = "https://router.huggingface.co/v1"
RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504, 524}
LOADING_HINTS = ("currently loading", "is loading", "model is loading", "warming up")


@dataclass
class HFInferenceTurnRecord:
    """Per-turn audit record persisted alongside the rollout."""

    step: int
    request_messages: list[dict[str, Any]]
    response_text: str
    response_tool_calls: list[dict[str, Any]]
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    latency_ms: int
    retries: int
    parse_error: str | None
    raw_response: dict[str, Any]
    thinking_content: str = ""
    # Forensics — populated even when the API call itself failed.
    truncated: bool = False
    provider: str | None = None
    api_error: dict[str, Any] | None = None


@dataclass
class HFInferencePolicy(_LLMPolicyBase):
    """OpenAI-compat policy targeting HF's hosted inference router.

    `model` accepts the bare HF model ID (auto-routed) or `model:provider`
    when you want to pin to a specific provider (e.g.
    `Qwen/Qwen3-32B-Instruct-2507:fireworks-ai`).

    `record_turns=True` (the default) preserves a `HFInferenceTurnRecord`
    for every step. The eval harness flushes these to disk so a human can
    review what the model actually emitted, byte-for-byte.
    """

    model: str = "Qwen/Qwen3-14B-Instruct-2507"
    temperature: float = 0.4
    top_p: float = 0.9
    # 16384 covers Qwen3 base variants whose <think> traces sometimes spike
    # to ~10K tokens before emitting the tool call. The provider strips
    # `<think>` from message.content for display but it still counts toward
    # max_tokens, so under-sizing this field shows up as 400 tool_use_failed
    # truncations rather than parse errors. Drop to ~1024 for non-reasoning
    # variants (Qwen3-Instruct-2507, GPT-4o, Sonnet) if cost matters.
    max_tokens: int = 16384
    api_base_url: str = HF_DEFAULT_BASE_URL
    api_key: str | None = None
    use_tool_calls: bool = True
    max_retries: int = 6
    initial_backoff_s: float = 2.0
    max_backoff_s: float = 60.0
    request_timeout_s: float = 120.0
    record_turns: bool = True
    name: str = "hf"
    turns: list[HFInferenceTurnRecord] = field(default_factory=list, init=False)
    _client: Any = field(default=None, init=False)
    _tool_schemas: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("HFInferencePolicy requires `pip install openai`.") from exc

        api_key = (
            self.api_key
            or os.getenv("HF_TOKEN")
            or os.getenv("HF_AUTH_TOKEN")
            or os.getenv("HUGGINGFACE_TOKEN")
            or os.getenv("HUGGING_FACE_HUB_TOKEN")
        )
        if not api_key:
            raise SystemExit(
                "HFInferencePolicy needs an HF token. Set one of HF_TOKEN / "
                "HF_AUTH_TOKEN / HUGGINGFACE_TOKEN, or pass api_key=...; "
                "this is the same token huggingface-cli login uses."
            )

        from dronecaptureops.agent.schemas import openai_tool_schemas

        self._client = OpenAI(
            base_url=self.api_base_url,
            api_key=api_key,
            timeout=self.request_timeout_s,
        )
        self._tool_schemas = openai_tool_schemas(self.env._tools)  # noqa: SLF001

    # --- Policy protocol -----------------------------------------------------

    def next_action(self, observation: DroneObservation, context: AgentContext) -> RawDroneAction:
        self._ensure_initialised()
        is_initial = len(self._messages) == 1
        from dronecaptureops.agent.messages import build_user_message

        self._messages.append(build_user_message(observation, is_initial=is_initial))

        request_messages = self._trimmed()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request_messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.use_tool_calls:
            kwargs["tools"] = self._tool_schemas
            kwargs["tool_choice"] = "auto"

        # Capture every API outcome — including hard failures — into a turn
        # record. Crashed cells with no JSONL trail are forensically useless;
        # we want every interaction reviewable post-hoc.
        try:
            response, retries, latency_ms = self._call_with_retry(**kwargs)
        except Exception as exc:  # noqa: BLE001 — we triage and re-raise
            api_error = _serialise_api_error(exc)
            failed_text = api_error.get("failed_generation") or ""
            error_msg = f"{type(exc).__name__}: {api_error.get('message', str(exc))[:200]}"
            if self.record_turns:
                self.turns.append(
                    HFInferenceTurnRecord(
                        step=len(self.turns) + 1,
                        request_messages=request_messages,
                        response_text=failed_text,
                        response_tool_calls=[],
                        finish_reason="api_error",
                        prompt_tokens=None,
                        completion_tokens=None,
                        total_tokens=None,
                        latency_ms=0,
                        retries=self.max_retries,
                        parse_error=error_msg,
                        raw_response={},
                        thinking_content="",
                        truncated=("tool_use_failed" in str(exc) or "max_tokens" in str(exc).lower()),
                        provider=None,
                        api_error=api_error,
                    )
                )
            # Surface as ActionValidationError so the rollout records a
            # parse-style failure for this turn instead of crashing the cell.
            raise ActionValidationError(error_msg) from exc

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        total_tokens = getattr(usage, "total_tokens", None) if usage else None

        # Persist what the model said so subsequent turns reflect reality.
        self._messages.append(_message_to_dict(message))

        parse_error: str | None = None
        thinking_content = ""
        action: RawDroneAction | None = None
        try:
            action, thinking_content = self._parse_response(message)
        except ActionValidationError as exc:
            parse_error = str(exc)

        if self.record_turns:
            tool_calls = getattr(message, "tool_calls", None) or []
            tool_calls_dict = [
                {
                    "id": getattr(call, "id", None),
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
            self.turns.append(
                HFInferenceTurnRecord(
                    step=len(self.turns) + 1,
                    request_messages=request_messages,
                    response_text=message.content or "",
                    response_tool_calls=tool_calls_dict,
                    finish_reason=finish_reason,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    retries=retries,
                    parse_error=parse_error,
                    raw_response=_serialise_response(response),
                    thinking_content=thinking_content,
                    truncated=(finish_reason == "length"),
                    provider=getattr(response, "model", None) or _provider_from_response(response),
                    api_error=None,
                )
            )

        if parse_error or action is None:
            raise ActionValidationError(parse_error or "parse failed")
        return action

    # --- helpers -------------------------------------------------------------

    def _call_with_retry(self, **kwargs: Any) -> tuple[Any, int, int]:
        """Call the OpenAI-compatible chat endpoint with retry + backoff.

        Retries on 429 (rate limit), 503 (model loading), and transient
        network errors. Hard-fails on auth errors and validation errors.
        """

        backoff = self.initial_backoff_s
        last_exc: Exception | None = None
        started = time.time()
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                return response, attempt, int((time.time() - started) * 1000)
            except Exception as exc:  # noqa: BLE001 - we triage status codes below
                last_exc = exc
                if not self._is_retryable(exc):
                    raise
                wait = min(backoff, self.max_backoff_s)
                LOG.warning(
                    "HF inference attempt %d/%d failed (%s); sleeping %.1fs",
                    attempt + 1,
                    self.max_retries,
                    _short_exc(exc),
                    wait,
                )
                time.sleep(wait)
                backoff = min(backoff * 2, self.max_backoff_s)
        raise RuntimeError(
            f"HF inference exhausted {self.max_retries} retries; last error: {_short_exc(last_exc)}"
        )

    def _is_retryable(self, exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        if status is None:
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
        message = str(exc).lower()
        if any(hint in message for hint in LOADING_HINTS):
            return True
        if status in RETRYABLE_STATUS_CODES:
            return True
        # Network-layer errors raised by httpx — keep going.
        if exc.__class__.__name__ in {"ReadTimeout", "ConnectError", "RemoteProtocolError"}:
            return True
        return False

    def _parse_response(self, message: Any) -> tuple[RawDroneAction, str]:
        """Return (action, thinking) extracted from a chat completion message.

        Priority differs from the obvious "trust tool_calls first" because
        provider-side tool-call extraction is the unreliable surface for
        reasoning models (Qwen3, DeepSeek-R1). The provider may strip
        `<think>` blocks inconsistently, leak whitespace into the
        `arguments` JSON string, or leave tool_calls empty entirely with
        the actual call inside `content`.

        Strategy:
          1. If `content` contains `<think>` or `<tool_call>` markers, parse
             content — that's where the model put the truth and it's
             markup we know how to handle.
          2. Else if `tool_calls` is populated, use it (the cheap path for
             non-reasoning models like Qwen3-Instruct-2507, GPT-4o, Sonnet).
          3. Else parse `content` as a generic text response.

        `thinking` is always populated when `<think>` blocks were stripped,
        regardless of which surface succeeded. Empty string otherwise.
        """

        content = message.content or ""
        tool_calls = getattr(message, "tool_calls", None)

        has_reasoning_markers = ("<think>" in content.lower()) or ("<tool_call>" in content.lower())

        if has_reasoning_markers:
            return parse_action_with_thinking(content)

        if tool_calls:
            payload = [
                {
                    "id": getattr(call, "id", None),
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ]
            return parse_action_with_thinking(payload)

        if not content.strip():
            raise ActionValidationError("model returned no tool_call and empty content")
        return parse_action_with_thinking(content)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _serialise_response(response: Any) -> dict[str, Any]:
    """Best-effort serialisation of an OpenAI ChatCompletion for audit logs."""

    if hasattr(response, "model_dump"):
        try:
            return response.model_dump()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(response, "to_dict"):
        try:
            return response.to_dict()  # type: ignore[no-any-return]
        except Exception:  # noqa: BLE001
            pass
    return {"repr": repr(response)}


def _short_exc(exc: Exception | None) -> str:
    if exc is None:
        return "unknown"
    text = repr(exc)
    return text if len(text) <= 200 else text[:197] + "..."


def _serialise_api_error(exc: Exception) -> dict[str, Any]:
    """Pull every available forensic field off an OpenAI/httpx exception.

    The HF router returns a structured 400 with `failed_generation` (the
    truncated text the model produced before the provider gave up parsing
    its tool call). We capture that, the status code, the error message,
    and any provider headers so post-hoc analysis can reconstruct exactly
    what went wrong without re-running the request.
    """

    info: dict[str, Any] = {
        "exc_type": type(exc).__name__,
        "message": str(exc)[:500],
    }
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status is not None:
        info["status_code"] = status

    # OpenAI BadRequestError exposes `body` as a dict with provider-specific keys.
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body) if isinstance(body.get("error"), dict) else body
        if isinstance(err, dict):
            for key in ("message", "code", "type", "failed_generation", "param"):
                if key in err:
                    info[key] = err[key]
    return info


def _provider_from_response(response: Any) -> str | None:
    """Best-effort: read the upstream provider name from response metadata.

    HF's router returns the underlying provider via the `model` field when
    pinned (e.g. `Qwen/Qwen3-32B:fireworks-ai`) and via response headers
    otherwise. Returns None when not surfaced.
    """

    headers = getattr(response, "_response_headers", None)
    if isinstance(headers, dict):
        for k in ("x-served-by", "x-provider", "x-routed-to"):
            if k in headers:
                return str(headers[k])
    return None


__all__ = ["HFInferencePolicy", "HFInferenceTurnRecord", "HF_DEFAULT_BASE_URL"]
