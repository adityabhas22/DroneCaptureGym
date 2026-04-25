"""Inference entry point: run one episode of any task with any policy.

Usage:
    python inference.py --task basic_thermal_survey --policy scripted
    python inference.py --task anomaly_confirmation --policy openai \\
        --model gpt-4o-mini
    python inference.py --task multi_anomaly_triage --policy anthropic \\
        --model claude-haiku-4-5-20251001
    python inference.py --task closeup_resolution_challenge --policy hf \\
        --model Qwen/Qwen2.5-7B-Instruct

The same harness drives every policy. Output is a structured JSON
trajectory plus a one-line summary on stderr; SFT data generation and
RL trainers will consume the same `RolloutResult` shape.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dronecaptureops.agent import (
    RandomPolicy,
    RolloutResult,
    RolloutRunner,
    ScriptedPolicy,
    trajectory_to_chat_messages,
)
from dronecaptureops.agent.policies import Policy
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


SUPPORTED_POLICIES = {"scripted", "random", "openai", "anthropic", "hf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one episode of DroneCaptureOps with any policy.")
    parser.add_argument("--task", default="basic_thermal_survey", choices=sorted(SOLAR_TASKS))
    parser.add_argument("--seed", type=int, default=int(os.getenv("DRONECAPTUREOPS_SEED", "7")))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--policy", default="scripted", choices=sorted(SUPPORTED_POLICIES))
    parser.add_argument("--model", default=None, help="Model ID for openai/anthropic/hf policies.")
    parser.add_argument("--api-base-url", default=os.getenv("OPENAI_API_BASE_URL") or os.getenv("API_BASE_URL"))
    parser.add_argument("--api-key", default=None, help="Override env-var API key.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens (chat completions).")
    parser.add_argument("--max-history-steps", type=int, default=12, help="How many user/assistant pairs to keep in context.")
    parser.add_argument("--no-tool-calls", action="store_true", help="Force JSON-text replies (skip native tool_calls).")
    parser.add_argument("--output", default=None, help="Optional JSONL path for the full trajectory record.")
    parser.add_argument("--messages-output", default=None, help="Optional JSONL path for the chat-format messages list.")
    parser.add_argument("--device", default="auto", help="Device for the local HF policy (auto, cpu, cuda).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass through to the HF model loader.")
    return parser.parse_args()


def build_policy(args: argparse.Namespace, env: DroneCaptureOpsEnvironment) -> Policy:
    if args.policy == "scripted":
        return ScriptedPolicy()
    if args.policy == "random":
        return RandomPolicy(seed=args.seed)

    if args.policy == "openai":
        from dronecaptureops.agent import OpenAIChatPolicy  # type: ignore[attr-defined]

        return OpenAIChatPolicy(
            env=env,
            task_id=args.task,
            model=args.model or "gpt-4o-mini",
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_history_steps=args.max_history_steps,
            use_tool_calls=not args.no_tool_calls,
        )

    if args.policy == "anthropic":
        from dronecaptureops.agent import AnthropicMessagesPolicy  # type: ignore[attr-defined]

        return AnthropicMessagesPolicy(
            env=env,
            task_id=args.task,
            model=args.model or "claude-haiku-4-5-20251001",
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_history_steps=args.max_history_steps,
            use_tool_calls=not args.no_tool_calls,
        )

    if args.policy == "hf":
        from dronecaptureops.agent import LocalHFPolicy  # type: ignore[attr-defined]

        return LocalHFPolicy(
            env=env,
            task_id=args.task,
            model=args.model or "Qwen/Qwen2.5-7B-Instruct",
            temperature=args.temperature,
            max_new_tokens=args.max_tokens,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            max_history_steps=args.max_history_steps,
        )

    raise SystemExit(f"unsupported policy: {args.policy}")


def write_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize(result: RolloutResult) -> dict:
    final_status = result.final_observation.get("checklist_status", {})
    return {
        "task_id": result.task_id,
        "policy": result.policy_name,
        "seed": result.seed,
        "steps": result.steps,
        "total_reward": round(result.total_reward, 4),
        "success": result.success,
        "complete": bool(final_status.get("complete")),
        "anomalies_detected": final_status.get("anomalies_detected") or [],
        "anomaly_rgb_pairs": final_status.get("anomaly_rgb_pairs") or {},
        "battery_pct_remaining": result.final_observation.get("telemetry", {}).get("battery", {}).get("level_pct"),
    }


def main() -> int:
    args = parse_args()
    env = DroneCaptureOpsEnvironment()
    policy = build_policy(args, env)
    runner = RolloutRunner(env=env)
    result = runner.run(policy, seed=args.seed, task_id=args.task, max_steps=args.max_steps)

    summary = summarize(result)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output:
        write_jsonl(Path(args.output), {"summary": summary, "result": result.model_dump(mode="json")})
        print(f"trajectory written to {args.output}", file=sys.stderr)

    if args.messages_output:
        messages = trajectory_to_chat_messages(result, use_tool_calls=not args.no_tool_calls)
        write_jsonl(
            Path(args.messages_output),
            {
                "task_id": result.task_id,
                "policy": result.policy_name,
                "seed": result.seed,
                "reward": result.total_reward,
                "success": result.success,
                "messages": messages,
            },
        )
        print(f"messages written to {args.messages_output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
