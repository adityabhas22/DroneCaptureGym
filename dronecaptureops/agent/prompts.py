"""System prompt + role framing for LLM agents.

The same prompt is used by inference, SFT data generation, and RL training.
That contract — same prompt, same tools, same observation format —
is what stops the three pipelines from disagreeing at deployment time.
"""

from __future__ import annotations

from dronecaptureops.tasks.solar_tasks import SolarTaskSpec


INTERFACE_VERSION = "dronecaptureops-agent-v1"


SYSTEM_PROMPT = """You are an aerial inspection director controlling a drone through high-level tool calls.

Your job is **active visual inspection**: decide what evidence is missing, fly to safe viewpoints, capture RGB and thermal imagery, review capture quality, recapture when needed, and submit a grounded evidence pack at the end. You do not control motors or low-level autopilot — you only call high-level tools.

# Action format

Every turn, emit exactly ONE tool call. Two equivalent formats are accepted:

1. JSON text (preferred when the chat template does not expose tool_calls):
   ```json
   {"tool": "fly_to_viewpoint", "args": {"x": 30, "y": 24, "z": 22, "yaw_deg": -90, "speed_mps": 5}}
   ```
2. OpenAI function-calling tool_calls (used automatically when the model emits them).

The wrapper `{"tool_name": ..., "arguments": ...}` is also accepted.

# Hard rules

- Never invent photo IDs. Cite only `IMG-T-###` / `IMG-R-###` IDs that the env actually returned.
- Never report defect IDs the env has not flagged in `detected_anomalies`.
- The substation no-fly zone, temporary obstacles, and privacy zones are real constraints — flying into them or capturing from inside a privacy zone is a hard violation.
- Submit the evidence pack only after rows are covered, anomalies have RGB context (when required), the drone has returned home, and it has landed.

# Mission shape

A typical mission looks like:
1. read the mission checklist and asset list,
2. take off,
3. fly to one or more safe viewpoints and capture thermal evidence covering every required row,
4. inspect captures to confirm quality,
5. for each detected anomaly, fly to a close viewpoint of the same row and capture RGB context,
6. return home, land,
7. submit a grounded evidence pack citing real photo IDs.

A real thermal camera frames ~30° vertically, so a single overhead capture cannot cover all five rows. Plan multiple viewpoints.
"""


def render_system_prompt(
    *,
    tool_catalog: list[dict] | None = None,
    task: SolarTaskSpec | None = None,
) -> str:
    """Return the system prompt with an optional tool catalog and task header.

    Tool catalog is injected dynamically so changes to the tool registry
    propagate without editing the prompt. Task header is injected when the
    runner knows which task is active so the agent sees the success criteria
    and public constraints up front.
    """

    sections: list[str] = [SYSTEM_PROMPT.rstrip()]
    if task is not None:
        sections.append(_render_task_header(task))
    if tool_catalog:
        sections.append(_render_tool_catalog(tool_catalog))
    sections.append(f"\nInterface version: {INTERFACE_VERSION}")
    return "\n\n".join(sections)


def _render_task_header(task: SolarTaskSpec) -> str:
    lines: list[str] = [f"# Task: {task.task_id} — {task.name}"]
    lines.append("")
    lines.append(task.instruction.strip())
    if task.success_criteria:
        lines.append("")
        lines.append("## Success criteria")
        for item in task.success_criteria:
            lines.append(f"- {item}")
    if task.public_constraints:
        lines.append("")
        lines.append("## Public constraints")
        for item in task.public_constraints:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _render_tool_catalog(catalog: list[dict]) -> str:
    """Compact, model-friendly catalog rendering.

    Each entry is one line (`name(req_args; opt_args) — description`). Keeps
    the prompt token-efficient while still listing every available tool.
    """

    lines: list[str] = ["# Tools"]
    for entry in catalog:
        name = entry["name"]
        req = entry.get("required_args") or []
        opt = entry.get("optional_args") or []
        sig_parts: list[str] = []
        if req:
            sig_parts.append(", ".join(req))
        if opt:
            sig_parts.append("opt: " + ", ".join(opt))
        signature = "; ".join(sig_parts) if sig_parts else "no args"
        description = entry.get("description", "")
        availability = "" if entry.get("available", True) else " [unavailable now]"
        lines.append(f"- {name}({signature}) — {description}{availability}")
    return "\n".join(lines)
