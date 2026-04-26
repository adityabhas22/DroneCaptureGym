"""OpsArena/OpenEnv submission inference entry point.

By default this script runs a reproducible three-task baseline and emits the
strict structured stdout format expected by the pre-submission validator:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

When API_BASE_URL, MODEL_NAME, and HF_TOKEN are present, the default policy is
OpenAI-compatible chat completions via the OpenAI client. For local smoke tests
without credentials, the script falls back to the deterministic scripted policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dronecaptureops.agent import (
    AgentContext,
    RandomPolicy,
    ScriptedPolicy,
    trajectory_to_chat_messages,
)
from dronecaptureops.agent.policies import Policy
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


ENV_NAME = "dronecaptureops-gym"
DEFAULT_TASKS = (
    "basic_thermal_survey",        # easy: coverage + grounded no-frills report
    "anomaly_confirmation",        # medium: detect and RGB-confirm a defect
    "audit_grade_strict_grounding", # hard: multi-issue strict report grounding
)
SUPPORTED_POLICIES = {"scripted", "random", "openai", "anthropic", "hf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one episode of DroneCaptureOps with any policy.")
    parser.add_argument("--task", default=None, choices=sorted(SOLAR_TASKS), help="Single task to run.")
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated task IDs for the baseline run. Ignored when --task is set.",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("DRONECAPTUREOPS_SEED", "7")))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--policy", default=_default_policy(), choices=sorted(SUPPORTED_POLICIES))
    parser.add_argument("--model", default=None, help="Model ID for openai/anthropic/hf policies.")
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL") or os.getenv("OPENAI_API_BASE_URL"))
    parser.add_argument("--api-key", default=None, help="Override env-var API key. Defaults to HF_TOKEN for OpenAI-compatible endpoints.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max generation tokens (chat completions).")
    parser.add_argument("--max-history-steps", type=int, default=12, help="How many user/assistant pairs to keep in context.")
    parser.add_argument("--no-tool-calls", action="store_true", help="Force JSON-text replies (skip native tool_calls).")
    parser.add_argument("--output", default=None, help="Optional JSONL path for the full trajectory record.")
    parser.add_argument("--messages-output", default=None, help="Optional JSONL path for the chat-format messages list.")
    parser.add_argument("--device", default="auto", help="Device for the local HF policy (auto, cpu, cuda).")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass through to the HF model loader.")
    return parser.parse_args()


def _default_policy() -> str:
    required = (os.getenv("API_BASE_URL"), os.getenv("MODEL_NAME"), os.getenv("HF_TOKEN"))
    return "openai" if all(required) else "scripted"


def _task_ids(args: argparse.Namespace) -> list[str]:
    if args.task:
        return [args.task]
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(SOLAR_TASKS))
    if unknown:
        raise SystemExit(f"unknown task ids: {unknown}")
    return tasks


def build_policy(args: argparse.Namespace, env: DroneCaptureOpsEnvironment, task_id: str) -> Policy:
    if args.policy == "scripted":
        return ScriptedPolicy()
    if args.policy == "random":
        return RandomPolicy(seed=args.seed)

    if args.policy == "openai":
        from dronecaptureops.agent import OpenAIChatPolicy  # type: ignore[attr-defined]

        model = args.model or os.getenv("MODEL_NAME")
        api_key = args.api_key or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
        if not args.api_base_url or not model or not api_key:
            raise SystemExit("openai policy requires API_BASE_URL, MODEL_NAME, and HF_TOKEN (or explicit CLI overrides)")
        return OpenAIChatPolicy(
            env=env,
            task_id=task_id,
            model=model,
            api_base_url=args.api_base_url,
            api_key=api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_history_steps=args.max_history_steps,
            use_tool_calls=not args.no_tool_calls,
        )

    if args.policy == "anthropic":
        from dronecaptureops.agent import AnthropicMessagesPolicy  # type: ignore[attr-defined]

        return AnthropicMessagesPolicy(
            env=env,
            task_id=task_id,
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
            task_id=task_id,
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


def _compact_action(action: RawDroneAction | dict[str, Any]) -> str:
    if isinstance(action, RawDroneAction):
        payload = {"tool_name": action.tool_name, "arguments": action.arguments}
    else:
        payload = {
            "tool_name": action.get("tool_name", "invalid"),
            "arguments": action.get("arguments", {}),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _error_value(error: str | None) -> str:
    if not error:
        return "null"
    return str(error).replace("\n", " ").replace("\r", " ").replace(" ", "_")


def _model_name(args: argparse.Namespace) -> str:
    if args.policy == "openai":
        return args.model or os.getenv("MODEL_NAME") or "unknown"
    return args.model or args.policy


def _start_line(task_id: str, model: str) -> str:
    return f"[START] task={task_id} env={ENV_NAME} model={model}"


def _step_line(step: int, action: RawDroneAction | dict[str, Any], observation: DroneObservation) -> str:
    done = "true" if observation.done else "false"
    return (
        f"[STEP] step={step} action={_compact_action(action)} "
        f"reward={float(observation.reward or 0.0):.2f} done={done} error={_error_value(observation.error)}"
    )


def _end_line(success: bool, steps: int, score: float, rewards: list[float]) -> str:
    success_text = "true" if success else "false"
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    return f"[END] success={success_text} steps={steps} score={score:.2f} rewards={reward_text}"


def _next_action(policy: Policy, observation: DroneObservation, context: AgentContext) -> tuple[RawDroneAction | dict[str, Any], str | None]:
    try:
        return policy.next_action(observation, context), None
    except ActionValidationError as exc:
        return {"tool_name": "invalid", "arguments": {}}, str(exc)


def run_episode(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    env = DroneCaptureOpsEnvironment()
    observation = env.reset(seed=args.seed, task=task_id)
    initial_observation = observation.model_dump(mode="json")
    policy = build_policy(args, env, task_id)
    context = AgentContext()
    rewards: list[float] = []
    steps = 0
    output_steps: list[dict[str, Any]] = []
    print(_start_line(task_id, _model_name(args)), flush=True)
    try:
        for step_number in range(1, args.max_steps + 1):
            action, parse_error = _next_action(policy, observation, context)
            next_observation = env.step(action)
            steps = step_number
            rewards.append(float(next_observation.reward or 0.0))
            print(_step_line(step_number, action, next_observation), flush=True)
            action_payload = action.model_dump(mode="json") if isinstance(action, RawDroneAction) else action
            output_steps.append(
                {
                    "step": step_number,
                    "observation": observation.model_dump(mode="json"),
                    "action": action_payload,
                    "next_observation": next_observation.model_dump(mode="json"),
                    "reward": float(next_observation.reward or 0.0),
                    "done": next_observation.done,
                    "error": next_observation.error,
                    "parse_error": parse_error,
                    "action_result": next_observation.action_result,
                }
            )
            if isinstance(action, RawDroneAction):
                context.append(action=action, observation=next_observation, action_result=next_observation.action_result)
            observation = next_observation
            if observation.done:
                break
        success = bool(observation.done and observation.checklist_status.complete)
        score = float(observation.reward or 0.0)
        return {
            "task_id": task_id,
            "policy": getattr(policy, "name", args.policy),
            "seed": args.seed,
            "steps": steps,
            "score": score,
            "success": success,
            "complete": observation.checklist_status.complete,
            "rewards": rewards,
            "initial_observation": initial_observation,
            "trajectory": output_steps,
            "final_observation": observation.model_dump(mode="json"),
        }
    finally:
        final_success = bool(observation.done and observation.checklist_status.complete)
        final_score = float(observation.reward or 0.0)
        print(_end_line(final_success, steps, final_score, rewards), flush=True)
        close = getattr(env, "close", None)
        if callable(close):
            close()


def main() -> int:
    args = parse_args()
    results = [run_episode(args, task_id) for task_id in _task_ids(args)]

    if args.output:
        write_jsonl(Path(args.output), {"results": results})
        print(f"trajectory written to {args.output}", file=sys.stderr)

    if args.messages_output:
        from dronecaptureops.agent import RolloutResult, RolloutStep

        # Preserve the old debug artifact shape for local SFT/debug workflows.
        message_records = []
        for result in results:
            rollout = RolloutResult(
                policy_name=result["policy"],
                seed=result["seed"],
                task_id=result["task_id"],
                success=result["success"],
                steps=result["steps"],
                total_reward=result["score"],
                reward_breakdown=result["final_observation"]["reward_breakdown"],
                trajectory=[
                    RolloutStep(
                        step=step["step"],
                        observation=step["observation"],
                        action=step["action"],
                        next_observation=step["next_observation"],
                        reward=step["reward"],
                        reward_breakdown={},
                        done=step["done"],
                        action_result=step["action_result"],
                        parse_error=step["parse_error"],
                    )
                    for step in result["trajectory"]
                ],
                initial_observation=result["initial_observation"],
                final_observation=result["final_observation"],
            )
            message_records.append(
                {
                    "task_id": result["task_id"],
                    "policy": result["policy"],
                    "seed": result["seed"],
                    "reward": result["score"],
                    "success": result["success"],
                    "messages": trajectory_to_chat_messages(rollout, use_tool_calls=not args.no_tool_calls),
                }
            )
        write_jsonl(Path(args.messages_output), {"runs": message_records})
        print(f"messages written to {args.messages_output}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
