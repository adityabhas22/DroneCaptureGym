"""Chat-format message construction.

The SFT data generator, the inference CLI, and the RL trainers all build
their prompts from the same primitives here. Sharing the message layout
means an SFT example, a GRPO completion, and a PPO rollout step are
visually identical to the model — a property the OpsArena/ClaimsGym
retros said is essential to avoid train/serve disagreement.
"""

from __future__ import annotations

import json
from typing import Any

from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.tasks.solar_tasks import SolarTaskSpec
from dronecaptureops.tools.registry import ToolRegistry

from dronecaptureops.agent.observation import render_observation, render_initial_observation
from dronecaptureops.agent.prompts import render_system_prompt


def build_system_message(
    *,
    registry: ToolRegistry,
    world,
    task: SolarTaskSpec | None = None,
) -> dict[str, str]:
    """One-shot system message including dynamic tool catalog and task header."""

    catalog = registry.catalog_as_json(world)
    return {
        "role": "system",
        "content": render_system_prompt(tool_catalog=catalog, task=task),
    }


def build_user_message(
    observation: DroneObservation,
    *,
    is_initial: bool,
) -> dict[str, str]:
    """Single user-turn payload: compact observation + reward delta."""

    body = render_initial_observation(observation) if is_initial else render_observation(observation)
    return {"role": "user", "content": body}


def build_assistant_message(
    action: RawDroneAction,
    *,
    use_tool_calls: bool = False,
) -> dict[str, Any]:
    """Assistant turn for SFT data: either a JSON-text tool call or
    OpenAI-style tool_calls array."""

    if use_tool_calls:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{action.tool_name}_{hash(json.dumps(action.arguments, sort_keys=True)) & 0xFFFFFF:06x}",
                    "type": "function",
                    "function": {
                        "name": action.tool_name,
                        "arguments": json.dumps(action.arguments, sort_keys=True),
                    },
                }
            ],
        }
    payload = {"tool": action.tool_name, "args": action.arguments}
    return {"role": "assistant", "content": json.dumps(payload, sort_keys=True)}


def build_tool_result_message(
    observation: DroneObservation,
    *,
    tool_call_id: str | None = None,
) -> dict[str, Any]:
    """Tool-result payload following an assistant tool_call.

    Mirrors the OpenAI `role: tool` schema; falls back to a `role: user`
    message when no tool_call_id is supplied (i.e. the JSON-text dialect).
    """

    body = render_observation(observation)
    if tool_call_id is None:
        return {"role": "user", "content": body}
    return {"role": "tool", "content": body, "tool_call_id": tool_call_id}


def trajectory_to_messages(
    *,
    initial_observation: DroneObservation,
    steps: list[dict[str, Any]],
    registry: ToolRegistry,
    world,
    task: SolarTaskSpec | None = None,
    use_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Convert a rollout trajectory into a chat-format messages list.

    `steps` is a list of `{action: RawDroneAction|dict, next_observation: DroneObservation}`
    in chronological order. The result is a [system, user, assistant, user, ...]
    sequence suitable for SFT, supervised eval, or RL replay.
    """

    messages: list[dict[str, Any]] = [
        build_system_message(registry=registry, world=world, task=task),
        build_user_message(initial_observation, is_initial=True),
    ]
    for step in steps:
        action = step["action"]
        if isinstance(action, dict):
            action = RawDroneAction(
                tool_name=action.get("tool_name") or action.get("tool", ""),
                arguments=action.get("arguments") or action.get("args", {}),
            )
        messages.append(build_assistant_message(action, use_tool_calls=use_tool_calls))
        messages.append(build_user_message(step["next_observation"], is_initial=False))
    return messages
