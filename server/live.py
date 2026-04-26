"""Live-session FastAPI routes for simulation and inference traces."""

from __future__ import annotations

import importlib
import json
import os
import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError, field_validator

from dronecaptureops.agent.policies import AgentContext, RandomPolicy, ScriptedPolicy
from dronecaptureops.agent.comparison import ComparisonRequest, ModelRunSpec, build_policy_for_spec, run_model_comparison
from dronecaptureops.agent.parser import parse_action
from dronecaptureops.core.constants import DEFAULT_DOMAIN
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.generation.suites import list_suites
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


router = APIRouter(prefix="/live", tags=["live"])
REPO_ROOT = Path(__file__).resolve().parents[1]
_LOG_LOCK = RLock()
_REDACTED = "***redacted***"


class LiveSessionCreate(BaseModel):
    """Request body for creating or resetting a live session."""

    session_id: str | None = None
    seed: int | None = None
    episode_id: str | None = None
    domain: str = DEFAULT_DOMAIN
    scenario_family: str | None = None
    task: str | None = None
    task_id: str | None = None

    @field_validator("session_id")
    @classmethod
    def session_id_must_be_route_safe(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("session_id cannot be empty")
        if "/" in normalized:
            raise ValueError("session_id cannot contain '/'")
        return normalized


class LiveStepRequest(BaseModel):
    """Manual action step request.

    The action is accepted as a dict so malformed RawDroneAction payloads can
    still be logged as parse errors instead of being rejected before replay.
    """

    action: dict[str, Any]


class LiveRunRequest(BaseModel):
    """Small synchronous policy-run request for lightweight inference demos."""

    policy: Literal["scripted", "random"] = "scripted"
    max_steps: int = Field(default=5, ge=1, le=100)
    seed: int = 0


class LiveModelRunRequest(BaseModel):
    """Run a model/policy spec through the current live session."""

    spec: ModelRunSpec
    max_steps: int = Field(default=20, ge=1, le=100)
    task_id: str | None = None
    scenario_family: str | None = None
    seed: int | None = None
    user_instruction: str | None = None


class LiveReplayRequest(BaseModel):
    """Replay tool calls from a rollout, trace, or SFT JSON/JSONL record."""

    source_path: str | None = None
    record_index: int = Field(default=0, ge=0)
    record: dict[str, Any] | None = None
    reset_session: bool = True
    max_steps: int | None = Field(default=None, ge=1, le=200)


class LiveEvent(BaseModel):
    """Replayable event emitted by a live session."""

    id: int
    session_id: str
    type: str
    step: int
    action: dict[str, Any] | None = None
    observation: dict[str, Any] | None = None
    reward: float | None = None
    reward_breakdown: dict[str, Any] = Field(default_factory=dict)
    action_result: dict[str, Any] = Field(default_factory=dict)
    scene: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    captures: list[dict[str, Any]] = Field(default_factory=list)
    done: bool = False
    parse_error: str | None = None
    message: str | None = None


@dataclass
class LiveSession:
    """In-memory live environment session."""

    session_id: str
    env: DroneCaptureOpsEnvironment
    observation: DroneObservation
    context: AgentContext = field(default_factory=AgentContext)
    events: list[LiveEvent] = field(default_factory=list)
    policy_cache: dict[str, Any] = field(default_factory=dict)


_SESSIONS: dict[str, LiveSession] = {}
_LOCK = RLock()


@router.post("/sessions")
def create_live_session(request: LiveSessionCreate | None = None) -> dict[str, Any]:
    """Create a live session and reset the environment."""

    body = request or LiveSessionCreate()
    _log_live("session.create.request", request=body.model_dump(mode="json"))
    session = _new_session(body)
    with _LOCK:
        _SESSIONS[session.session_id] = session
    _log_live("session.create.response", session_id=session.session_id, state=_observation_summary(session.observation))
    return _session_payload(session)


@router.get("/sessions")
def list_live_sessions() -> dict[str, Any]:
    """List in-memory live sessions for the browser console."""

    with _LOCK:
        return {
            "sessions": [
                {
                    "session_id": session.session_id,
                    "step_count": session.observation.metadata.get("step_count", 0),
                    "task_id": session.observation.metadata.get("task_id"),
                    "scenario_family": session.observation.metadata.get("scenario_family"),
                    "scenario_seed": session.observation.metadata.get("scenario_seed"),
                    "done": session.observation.done,
                    "event_count": len(session.events),
                }
                for session in _SESSIONS.values()
            ]
        }


@router.post("/sessions/{session_id}/reset")
def reset_live_session(session_id: str, request: LiveSessionCreate | None = None) -> dict[str, Any]:
    """Reset an existing live session ID, replacing any previous state."""

    body = (request or LiveSessionCreate()).model_copy(update={"session_id": session_id})
    _log_live("session.reset.request", session_id=session_id, request=body.model_dump(mode="json"))
    session = _new_session(body)
    with _LOCK:
        _SESSIONS[session.session_id] = session
    _log_live("session.reset.response", session_id=session.session_id, state=_observation_summary(session.observation))
    return _session_payload(session)


@router.get("/sessions/{session_id}")
def get_live_session(session_id: str) -> dict[str, Any]:
    """Return the current visible state and latest scene snapshot."""

    with _LOCK:
        session = _get_session(session_id)
        return _session_payload(session)


@router.post("/sessions/{session_id}/step")
def step_live_session(session_id: str, request: LiveStepRequest) -> dict[str, Any]:
    """Apply one RawDroneAction-compatible tool call to a live session."""

    _log_live("session.step.request", session_id=session_id, action=request.action)
    with _LOCK:
        session = _get_session(session_id)
        session.policy_cache.clear()
        event = _apply_action(session, request.action)
        _log_live(
            "session.step.response",
            session_id=session.session_id,
            event_id=event.id,
            state=_observation_summary(session.observation),
        )
        return {
            "session_id": session.session_id,
            "event": event.model_dump(mode="json"),
            "observation": _dump_model(session.observation),
            "state": _dump_model(session.env.state),
            "scene": _scene_snapshot(session.env, session.observation),
        }


@router.post("/sessions/{session_id}/run")
def run_live_session(session_id: str, request: LiveRunRequest) -> dict[str, Any]:
    """Run a lightweight synchronous policy loop and append replay events."""

    _log_live("session.run.request", session_id=session_id, request=request.model_dump(mode="json"))
    with _LOCK:
        session = _get_session(session_id)
        session.policy_cache.clear()
        policy = ScriptedPolicy() if request.policy == "scripted" else RandomPolicy(seed=request.seed)
        result_events: list[LiveEvent] = []
        for _ in range(request.max_steps):
            if session.observation.done:
                break
            try:
                action: RawDroneAction | dict[str, Any] = policy.next_action(session.observation, session.context)
                parse_error = None
            except ActionValidationError as exc:
                action = {"tool_name": "invalid", "arguments": {}}
                parse_error = str(exc)
            result_events.append(_apply_action(session, action, parse_error=parse_error))
        _log_live(
            "session.run.response",
            session_id=session.session_id,
            policy=request.policy,
            emitted_events=len(result_events),
            state=_observation_summary(session.observation),
        )
        return {
            "session_id": session.session_id,
            "policy": request.policy,
            "events": [event.model_dump(mode="json") for event in result_events],
            "observation": _dump_model(session.observation),
            "state": _dump_model(session.env.state),
            "scene": _scene_snapshot(session.env, session.observation),
        }


@router.post("/sessions/{session_id}/run_model")
def run_live_model_session(session_id: str, request: LiveModelRunRequest) -> dict[str, Any]:
    """Run a chat/model spec in the current live session and stream tool-call events."""

    _log_live(
        "model.run.request",
        session_id=session_id,
        spec=request.spec.model_dump(mode="json"),
        max_steps=request.max_steps,
        task_id=request.task_id,
        scenario_family=request.scenario_family,
        seed=request.seed,
        user_instruction_length=len(request.user_instruction or ""),
        user_instruction_preview=(request.user_instruction or "")[:500],
    )
    with _LOCK:
        session = _get_session(session_id)
        policy_key = _policy_cache_key(request)
        policy = session.policy_cache.get(policy_key)
        try:
            if policy is None:
                policy = build_policy_for_spec(
                    request.spec,
                    env=session.env,
                    task_id=request.task_id or _task_id_from_observation(session.observation),
                    seed=request.seed or _seed_from_observation(session.observation),
                    scenario_family=request.scenario_family or _scenario_family_from_observation(session.observation),
                    user_instruction=request.user_instruction,
                )
                session.policy_cache[policy_key] = policy
                _log_live("model.policy.cache_miss", session_id=session.session_id, policy=request.spec.name)
            else:
                _log_live("model.policy.cache_hit", session_id=session.session_id, policy=request.spec.name)
        except Exception as exc:  # noqa: BLE001 - diagnostics should preserve construction failures
            _log_live(
                "model.run.build_error",
                session_id=session.session_id,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            raise
        result_events: list[LiveEvent] = []
        for iteration in range(1, request.max_steps + 1):
            if session.observation.done:
                _log_live(
                    "model.run.stop_done",
                    session_id=session.session_id,
                    iteration=iteration,
                    state=_observation_summary(session.observation),
                )
                break
            try:
                _log_live(
                    "model.next_action.start",
                    session_id=session.session_id,
                    iteration=iteration,
                    policy=request.spec.name,
                    state=_observation_summary(session.observation),
                )
                action: RawDroneAction | dict[str, Any] = policy.next_action(session.observation, session.context)
                parse_error = None
                _log_live(
                    "model.next_action.success",
                    session_id=session.session_id,
                    iteration=iteration,
                    action=_action_payload(action),
                )
            except ActionValidationError as exc:
                action = {"tool_name": "invalid", "arguments": {}}
                parse_error = str(exc)
                _log_live(
                    "model.next_action.parse_error",
                    session_id=session.session_id,
                    iteration=iteration,
                    parse_error=parse_error,
                )
            except Exception as exc:  # noqa: BLE001 - live sessions should preserve model failures as events
                _log_live(
                    "model.next_action.error",
                    session_id=session.session_id,
                    iteration=iteration,
                    error_type=type(exc).__name__,
                    error=str(exc),
                    traceback=traceback.format_exc(),
                )
                result_events.append(
                    _append_event(
                        session,
                        "model-error",
                        parse_error=f"{type(exc).__name__}: {exc}",
                        message="model run failed before emitting a valid action",
                    )
                )
                break
            result_events.append(_apply_action(session, action, parse_error=parse_error))
        _log_live(
            "model.run.response",
            session_id=session.session_id,
            policy=request.spec.name,
            emitted_events=len(result_events),
            state=_observation_summary(session.observation),
        )
        return {
            "session_id": session.session_id,
            "policy": request.spec.name,
            "events": [event.model_dump(mode="json") for event in result_events],
            "observation": _dump_model(session.observation),
            "state": _dump_model(session.env.state),
            "scene": _scene_snapshot(session.env, session.observation),
        }


@router.post("/sessions/{session_id}/replay")
def replay_live_session_record(session_id: str, request: LiveReplayRequest) -> dict[str, Any]:
    """Replay an existing rollout/SFT record into a live event stream."""

    _log_live("replay.request", session_id=session_id, request=request.model_dump(mode="json"))
    record = _load_replay_record(request)
    actions, parse_errors = _actions_from_record(record)
    if request.max_steps is not None:
        actions = actions[:request.max_steps]

    reset = _reset_request_from_record(record, session_id=session_id)
    with _LOCK:
        if request.reset_session:
            session = _new_session(reset)
            _SESSIONS[session.session_id] = session
        else:
            session = _get_session(session_id)
            session.policy_cache.clear()
        result_events: list[LiveEvent] = []
        for action in actions:
            if session.observation.done:
                break
            result_events.append(_apply_action(session, action))
        if parse_errors:
            result_events.append(
                _append_event(
                    session,
                    "replay-parse-warning",
                    parse_error="; ".join(parse_errors[:5]),
                    message=f"{len(parse_errors)} assistant action(s) could not be parsed",
                )
            )
        _log_live(
            "replay.response",
            session_id=session.session_id,
            source=request.source_path,
            actions_requested=len(actions),
            events_emitted=len(result_events),
            parse_errors=parse_errors,
            state=_observation_summary(session.observation),
        )
        return {
            "session_id": session.session_id,
            "source": request.source_path,
            "actions_replayed": len(result_events) - (1 if parse_errors else 0),
            "parse_errors": parse_errors,
            "events": [event.model_dump(mode="json") for event in result_events],
            "observation": _dump_model(session.observation),
            "state": _dump_model(session.env.state),
            "scene": _scene_snapshot(session.env, session.observation),
        }


@router.post("/compare")
def compare_live_models(request: ComparisonRequest) -> dict[str, Any]:
    """Run comparable model specs on the same task/seed prompt contract."""

    _log_live("compare.request", request=request.model_dump(mode="json"))
    result = run_model_comparison(request)
    _log_live(
        "compare.response",
        summary_count=len(result.summaries),
        summaries=[
            {
                "name": summary.name,
                "policy": summary.policy,
                "steps": summary.steps,
                "success": summary.success,
                "final_reward": summary.final_reward,
            }
            for summary in result.summaries
        ],
    )
    return result.model_dump(mode="json")


@router.get("/sessions/{session_id}/events")
def replay_live_events(
    session_id: str,
    since_id: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    """Return a JSON replay of events emitted after ``since_id``."""

    with _LOCK:
        session = _get_session(session_id)
        events = [event for event in session.events if event.id > since_id]
        return {
            "session_id": session.session_id,
            "events": [event.model_dump(mode="json") for event in events],
        }


@router.get("/sessions/{session_id}/events/stream")
def stream_live_events(
    session_id: str,
    since_id: int = Query(default=0, ge=0),
) -> StreamingResponse:
    """Replay current session events as Server-Sent Events.

    This is intentionally a finite replay stream for now; callers can reconnect
    with the latest event ID instead of requiring a background worker.
    """

    with _LOCK:
        session = _get_session(session_id)
        events = [event for event in session.events if event.id > since_id]
    return StreamingResponse(
        (_format_sse(event) for event in events),
        media_type="text/event-stream",
    )


@router.get("/tasks")
def list_live_tasks(limit: int | None = Query(default=None, ge=1)) -> dict[str, Any]:
    """List deterministic task specs that can seed live sessions."""

    task_specs = sorted(SOLAR_TASKS.values(), key=lambda task: task.task_id)
    if limit is not None:
        task_specs = task_specs[:limit]
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "name": task.name,
                "tags": list(task.task_tags),
                "required_rows": list(task.required_rows),
                "max_steps": task.max_steps,
            }
            for task in task_specs
        ]
    }


@router.get("/suites")
def list_live_suites() -> dict[str, Any]:
    """List scenario suites available for demo/live resets."""

    return {
        "suites": [
            {
                "name": suite.name,
                "purpose": suite.purpose,
                "heldout": suite.heldout,
                "families": list(suite.families),
                "seeds": list(suite.seeds),
                "episode_count": len(suite.episodes),
                "episodes": [
                    {
                        "episode_id": episode.episode_id,
                        "scenario_family": episode.scenario_family,
                        "seed": episode.seed,
                        "task_id": episode.task_id,
                        "tags": list(episode.tags),
                    }
                    for episode in suite.episodes[:10]
                ],
            }
            for suite in list_suites()
        ]
    }


@router.get("/diagnostics/logs")
def read_live_diagnostics_logs(limit: int = Query(default=200, ge=1, le=1000)) -> dict[str, Any]:
    """Return recent JSONL diagnostics entries for backend debugging."""

    path = _live_log_path()
    if not path.exists():
        return {"path": str(path), "entries": []}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"event_type": "log.decode_error", "raw": line})
    return {"path": str(path), "entries": entries}


def _load_replay_record(request: LiveReplayRequest) -> dict[str, Any]:
    if request.record is not None:
        return request.record
    if not request.source_path:
        raise HTTPException(status_code=422, detail="provide either record or source_path")

    path = _resolve_replay_path(request.source_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"replay source not found: {request.source_path}")
    if path.suffix == ".jsonl":
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if request.record_index >= len(lines):
            raise HTTPException(status_code=404, detail=f"record_index {request.record_index} outside JSONL length {len(lines)}")
        return json.loads(lines[request.record_index])

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if request.record_index >= len(payload):
            raise HTTPException(status_code=404, detail=f"record_index {request.record_index} outside JSON length {len(payload)}")
        record = payload[request.record_index]
    else:
        record = payload
    if not isinstance(record, dict):
        raise HTTPException(status_code=422, detail="replay record must be a JSON object")
    return record


def _resolve_replay_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="replay source must live under the repository root") from exc
    return resolved


def _actions_from_record(record: dict[str, Any]) -> tuple[list[RawDroneAction], list[str]]:
    if result := record.get("result"):
        if isinstance(result, dict):
            return _actions_from_rollout_payload(result)
    if "trajectory" in record:
        return _actions_from_rollout_payload(record)
    if "episode_steps" in record and isinstance(record["episode_steps"], list):
        return _actions_from_steps(record["episode_steps"])
    if "messages" in record:
        return _actions_from_messages(record["messages"])
    if "actions" in record and isinstance(record["actions"], list):
        actions: list[RawDroneAction] = []
        errors: list[str] = []
        for index, payload in enumerate(record["actions"], start=1):
            try:
                actions.append(parse_action(payload))
            except ActionValidationError as exc:
                errors.append(f"action {index}: {exc}")
        return actions, errors
    raise HTTPException(status_code=422, detail="record does not contain result, trajectory, episode_steps, messages, or actions")


def _actions_from_rollout_payload(payload: dict[str, Any]) -> tuple[list[RawDroneAction], list[str]]:
    trajectory = payload.get("trajectory") or []
    if not isinstance(trajectory, list):
        raise HTTPException(status_code=422, detail="rollout trajectory must be a list")
    return _actions_from_steps(trajectory)


def _actions_from_steps(steps: list[Any]) -> tuple[list[RawDroneAction], list[str]]:
    actions: list[RawDroneAction] = []
    errors: list[str] = []
    for index, step in enumerate(steps, start=1):
        action_payload = step.get("action") if isinstance(step, dict) else None
        if action_payload is None:
            errors.append(f"step {index}: missing action")
            continue
        try:
            actions.append(parse_action(action_payload))
        except ActionValidationError as exc:
            errors.append(f"step {index}: {exc}")
    return actions, errors


def _actions_from_messages(messages: Any) -> tuple[list[RawDroneAction], list[str]]:
    if not isinstance(messages, list):
        raise HTTPException(status_code=422, detail="messages must be a list")
    actions: list[RawDroneAction] = []
    errors: list[str] = []
    assistant_index = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        assistant_index += 1
        payload = message.get("tool_calls") or message.get("content") or ""
        try:
            actions.append(parse_action(payload))
        except ActionValidationError as exc:
            errors.append(f"assistant {assistant_index}: {exc}")
    return actions, errors


def _reset_request_from_record(record: dict[str, Any], *, session_id: str) -> LiveSessionCreate:
    result = record.get("result") if isinstance(record.get("result"), dict) else record
    metadata = result.get("final_observation", {}).get("metadata", {}) if isinstance(result, dict) else {}
    return LiveSessionCreate(
        session_id=session_id,
        seed=record.get("seed") or result.get("seed") or metadata.get("scenario_seed"),
        scenario_family=record.get("scenario_family") or result.get("scenario_family") or metadata.get("scenario_family"),
        task_id=record.get("task_id") or result.get("task_id") or metadata.get("task_id"),
    )


def _task_id_from_observation(observation: DroneObservation) -> str | None:
    value = observation.metadata.get("task_id")
    return str(value) if value else None


def _seed_from_observation(observation: DroneObservation) -> int | None:
    value = observation.metadata.get("scenario_seed")
    return int(value) if isinstance(value, int | float) else None


def _scenario_family_from_observation(observation: DroneObservation) -> str | None:
    value = observation.metadata.get("scenario_family")
    return str(value) if value else None


def _new_session(request: LiveSessionCreate) -> LiveSession:
    session_id = request.session_id or uuid4().hex
    env = DroneCaptureOpsEnvironment()
    observation = env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        domain=request.domain,
        scenario_family=request.scenario_family,
        task=request.task,
        task_id=request.task_id,
    )
    session = LiveSession(session_id=session_id, env=env, observation=observation)
    _append_event(session, "session-created", observation=observation, message="session created")
    return session


def _live_log_path() -> Path:
    configured = os.getenv("DRONECAPTUREOPS_LIVE_LOG_PATH")
    path = Path(configured).expanduser() if configured else REPO_ROOT / "artifacts" / "live" / "live-server.jsonl"
    return path if path.is_absolute() else REPO_ROOT / path


def _log_live(event_type: str, **payload: Any) -> None:
    entry = {
        "ts": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        **_redact(payload),
    }
    path = _live_log_path()
    try:
        with _LOG_LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, sort_keys=True, default=str) + "\n")
    except OSError:
        # Diagnostics must never break the live control path.
        return


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str.lower() in {
                "api_key",
                "authorization",
                "hf_token",
                "openai_api_key",
                "openai_key",
                "gpt_api_key",
                "gpt_token",
                "token",
                "password",
            }:
                redacted[key_str] = _REDACTED
            else:
                redacted[key_str] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if hasattr(value, "model_dump"):
        return _redact(value.model_dump(mode="json"))
    return value


def _policy_cache_key(request: LiveModelRunRequest) -> str:
    return json.dumps(
        {
            "spec": request.spec.model_dump(mode="json"),
            "task_id": request.task_id,
            "scenario_family": request.scenario_family,
            "seed": request.seed,
            "user_instruction": request.user_instruction,
        },
        sort_keys=True,
        default=str,
    )


def _observation_summary(observation: DroneObservation) -> dict[str, Any]:
    telemetry = observation.telemetry
    return {
        "episode_id": observation.metadata.get("episode_id"),
        "task_id": observation.metadata.get("task_id"),
        "scenario_family": observation.metadata.get("scenario_family"),
        "scenario_seed": observation.metadata.get("scenario_seed"),
        "step_count": observation.metadata.get("step_count"),
        "done": observation.done,
        "reward": observation.reward,
        "system_message": observation.system_message,
        "error": observation.error,
        "pose": telemetry.pose.model_dump(mode="json") if telemetry else None,
        "battery_pct": telemetry.battery.level_pct if telemetry else None,
        "mode": telemetry.autopilot.mode if telemetry else None,
        "capture_count": len(observation.capture_log),
        "warning_count": len(observation.warnings),
    }


def _get_session(session_id: str) -> LiveSession:
    try:
        return _SESSIONS[session_id]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown live session: {session_id}") from exc


def _apply_action(
    session: LiveSession,
    action: RawDroneAction | dict[str, Any],
    *,
    parse_error: str | None = None,
) -> LiveEvent:
    typed_action: RawDroneAction | None = action if isinstance(action, RawDroneAction) else None
    if typed_action is None and parse_error is None:
        try:
            typed_action = RawDroneAction.model_validate(action)
        except ValidationError as exc:
            parse_error = str(exc)

    event_action = typed_action if typed_action is not None else action
    _log_live(
        "env.step.start",
        session_id=session.session_id,
        action=_action_payload(event_action),
        parse_error=parse_error,
        state=_observation_summary(session.observation),
    )
    _append_event(
        session,
        "action-start",
        action=event_action,
        parse_error=parse_error,
        message="action started",
    )
    observation = session.env.step(typed_action if typed_action is not None else action)
    session.observation = observation
    if typed_action is not None:
        session.context.append(
            action=typed_action,
            observation=observation,
            action_result=observation.action_result,
        )
    event = _append_event(
        session,
        "action-result",
        observation=observation,
        action=event_action,
        action_result=observation.action_result,
        parse_error=parse_error,
        message=observation.system_message,
    )
    _log_live(
        "env.step.result",
        session_id=session.session_id,
        action=_action_payload(event_action),
        parse_error=parse_error,
        action_result=observation.action_result,
        state=_observation_summary(observation),
    )
    return event


def _append_event(
    session: LiveSession,
    event_type: str,
    *,
    observation: DroneObservation | None = None,
    action: RawDroneAction | dict[str, Any] | None = None,
    action_result: dict[str, Any] | None = None,
    parse_error: str | None = None,
    message: str | None = None,
) -> LiveEvent:
    obs = observation or session.observation
    obs_payload = _dump_model(obs)
    event = LiveEvent(
        id=len(session.events) + 1,
        session_id=session.session_id,
        type=event_type,
        step=int(obs_payload.get("metadata", {}).get("step_count", 0)),
        action=_action_payload(action),
        observation=obs_payload,
        reward=float(obs.reward or 0.0),
        reward_breakdown=_dump_model(obs.reward_breakdown),
        action_result=action_result or dict(obs.action_result),
        scene=_scene_snapshot(session.env, obs),
        warnings=list(obs.warnings),
        captures=[_dump_model(capture) for capture in obs.capture_log],
        done=obs.done,
        parse_error=parse_error,
        message=message,
    )
    session.events.append(event)
    _log_live(
        "live.event",
        session_id=session.session_id,
        event_id=event.id,
        live_event_type=event_type,
        step=event.step,
        action=event.action,
        reward=event.reward,
        done=event.done,
        parse_error=event.parse_error,
        message=event.message,
        warning_count=len(event.warnings),
        capture_count=len(event.captures),
        action_result=event.action_result,
        scene_schema=event.scene.get("schema_version") or event.scene.get("source"),
    )
    return event


def _session_payload(session: LiveSession) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "observation": _dump_model(session.observation),
        "state": _dump_model(session.env.state),
        "scene": _scene_snapshot(session.env, session.observation),
        "event_count": len(session.events),
        "latest_event_id": session.events[-1].id if session.events else None,
    }


def _scene_snapshot(env: DroneCaptureOpsEnvironment, observation: DroneObservation) -> dict[str, Any]:
    rich_snapshot = _rich_scene_snapshot(env, observation)
    if rich_snapshot is not None:
        return rich_snapshot
    return _geometry_scene_snapshot(observation)


def _rich_scene_snapshot(env: DroneCaptureOpsEnvironment, observation: DroneObservation) -> dict[str, Any] | None:
    """Use an optional rich_sim scene module when present.

    Expected simple function API: ``build_scene_snapshot(world, observation)``.
    Keyword-only variants are also accepted so the future scene module can stay
    decoupled from FastAPI.
    """

    for module_name in ("dronecaptureops.rich_sim.scene", "dronecaptureops.rich_sim.scenes"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            continue
        try:
            if builder := getattr(module, "build_scene_from_observation", None):
                snapshot = builder(observation)
            elif builder := getattr(module, "build_scene_snapshot", None):
                snapshot = builder(env.debug_world, observation)
            elif builder := getattr(module, "scene_snapshot", None):
                snapshot = builder(world=env.debug_world, observation=observation, env=env)
            elif builder := getattr(module, "snapshot_scene", None):
                snapshot = builder(observation)
            else:
                continue
        except Exception:
            return None
        return _jsonable(snapshot)
    return None


def _geometry_scene_snapshot(observation: DroneObservation) -> dict[str, Any]:
    telemetry = observation.telemetry
    site_map = observation.site_map
    captures = observation.capture_log
    return {
        "source": "geometry_fallback",
        "episode": dict(observation.metadata),
        "drone": {
            "pose": _dump_model(telemetry.pose) if telemetry else None,
            "gimbal": _dump_model(telemetry.gimbal) if telemetry else None,
            "camera": _dump_model(telemetry.camera) if telemetry else None,
            "mode": telemetry.autopilot.mode if telemetry else None,
            "battery_pct": telemetry.battery.level_pct if telemetry else None,
        },
        "home": _dump_model(site_map.home) if site_map else None,
        "assets": [
            {
                "asset_id": asset.asset_id,
                "asset_type": asset.asset_type,
                "label": asset.label,
                "geometry": _dump_model(asset.geometry),
                "required_modalities": list(asset.required_modalities),
            }
            for asset in (site_map.assets if site_map else [])
        ],
        "airspace_zones": [
            {
                "zone_id": zone.zone_id,
                "label": zone.label,
                "zone_type": zone.zone_type,
                "constraint_level": zone.constraint_level,
                "bounds": {
                    "min_x": zone.min_x,
                    "min_y": zone.min_y,
                    "max_x": zone.max_x,
                    "max_y": zone.max_y,
                    "min_altitude_m": zone.min_altitude_m,
                    "max_altitude_m": zone.max_altitude_m,
                },
            }
            for zone in (site_map.airspace_zones if site_map else [])
        ],
        "viewpoints": [
            {
                "viewpoint_id": viewpoint.viewpoint_id,
                "label": viewpoint.label,
                "pose": _dump_model(viewpoint.pose),
                "asset_ids": list(viewpoint.asset_ids),
                "standoff_bucket": viewpoint.standoff_bucket,
                "suitable_modalities": list(viewpoint.suitable_modalities),
            }
            for viewpoint in (site_map.viewpoints if site_map else [])
        ],
        "captures": [
            {
                "photo_id": capture.photo_id,
                "sensor": capture.sensor,
                "pose": _dump_model(capture.pose),
                "asset_ids": list(capture.asset_ids),
                "targets_visible": list(capture.targets_visible),
                "quality_score": capture.quality_score,
                "warnings": list(capture.warnings),
            }
            for capture in captures
        ],
        "reward": float(observation.reward or 0.0),
        "done": observation.done,
    }


def _format_sse(event: LiveEvent) -> str:
    payload = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
    return f"id: {event.id}\nevent: {event.type}\ndata: {payload}\n\n"


def _action_payload(action: RawDroneAction | dict[str, Any] | None) -> dict[str, Any] | None:
    if action is None:
        return None
    return _dump_model(action) if isinstance(action, RawDroneAction) else _jsonable(action)


def _dump_model(value: Any) -> dict[str, Any]:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else _jsonable(value)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return json.loads(json.dumps(value, default=str))
