"""OpenEnv/OpsArena-style submission inference entry point.

Structured stdout:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from dronecaptureops.agent import AgentContext, RandomPolicy, ScriptedPolicy, trajectory_to_chat_messages
from dronecaptureops.agent.policies import Policy
from dronecaptureops.agent.rollout import RolloutResult, RolloutStep
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.errors import ActionValidationError
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from dronecaptureops.tasks.solar_tasks import SOLAR_TASKS


ENV_NAME = "dronecaptureops-gym"

DEFAULT_TASKS = (
    "basic_thermal_survey",
    "anomaly_confirmation",
    "audit_grade_strict_grounding",
)

SUPPORTED_POLICIES = {"scripted", "random", "openai", "anthropic", "hf"}


def _default_policy() -> str:
    required = (os.getenv("API_BASE_URL"), os.getenv("MODEL_NAME"), os.getenv("HF_TOKEN"))
    return "openai" if all(required) else "scripted"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DroneCaptureOps submission inference.")
    parser.add_argument("--task", default=None, choices=sorted(SOLAR_TASKS), help="Single task to run.")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS), help="Comma-separated task IDs.")
    parser.add_argument("--seed", type=int, default=int(os.getenv("DRONECAPTUREOPS_SEED", "7")))
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--policy", default=_default_policy(), choices=sorted(SUPPORTED_POLICIES))
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base-url", default=os.getenv("API_BASE_URL") or os.getenv("OPENAI_API_BASE_URL"))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-history-steps", type=int, default=12)
    parser.add_argument("--no-tool-calls", action="store_true")
    parser.add_argument("--output", default=None, help="Optional JSONL trajectory output.")
    parser.add_argument("--messages-output", default=None, help="Optional JSON messages output.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


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
            raise SystemExit("openai policy requires API_BASE_URL, MODEL_NAME, and HF_TOKEN")
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


def write_jsonl(path: Path, payload: dict[str, Any]) -> None:
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
    return str(error).replace("\n", " ").replace("\r", " ")


def _model_name(args: argparse.Namespace) -> str:
    if args.policy == "openai":
        return args.model or os.getenv("MODEL_NAME") or "unknown"
    return args.model or args.policy


def _start_line(task_id: str, model: str) -> str:
    return f"[START] task={task_id} env={ENV_NAME} model={model}"


def _step_line(step: int, action: RawDroneAction | dict[str, Any], observation: DroneObservation) -> str:
    done = "true" if observation.done else "false"
    reward = float(observation.reward or 0.0)
    return (
        f"[STEP] step={step} action={_compact_action(action)} "
        f"reward={reward:.2f} done={done} error={_error_value(observation.error)}"
    )


def _end_line(success: bool, steps: int, score: float, rewards: list[float]) -> str:
    success_text = "true" if success else "false"
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    return f"[END] success={success_text} steps={steps} score={score:.2f} rewards={reward_text}"


def _next_action(
    policy: Policy,
    observation: DroneObservation,
    context: AgentContext,
) -> tuple[RawDroneAction | dict[str, Any], str | None]:
    try:
        return policy.next_action(observation, context), None
    except ActionValidationError as exc:
        return {"tool_name": "invalid", "arguments": {}}, str(exc)


def _rollout_result_from_run(run: dict[str, Any]) -> RolloutResult:
    return RolloutResult(
        policy_name=run["policy"],
        seed=run["seed"],
        task_id=run["task_id"],
        success=run["success"],
        steps=run["steps"],
        total_reward=run["score"],
        reward_breakdown=run["final_observation"].get("reward_breakdown", {}),
        trajectory=[RolloutStep.model_validate(step) for step in run["trajectory"]],
        initial_observation=run["initial_observation"],
        final_observation=run["final_observation"],
    )


def run_episode(args: argparse.Namespace, task_id: str) -> dict[str, Any]:
    env = DroneCaptureOpsEnvironment()
    observation = env.reset(seed=args.seed, task=task_id)
    initial_observation = observation.model_dump(mode="json")
    policy = build_policy(args, env, task_id)
    context = AgentContext()

    rewards: list[float] = []
    output_steps: list[dict[str, Any]] = []
    steps = 0

    print(_start_line(task_id, _model_name(args)), flush=True)

    for step_number in range(1, args.max_steps + 1):
        action, parse_error = _next_action(policy, observation, context)
        next_observation = env.step(action)
        steps = step_number
        reward = float(next_observation.reward or 0.0)
        rewards.append(reward)

        print(_step_line(step_number, action, next_observation), flush=True)

        action_payload = action.model_dump(mode="json") if isinstance(action, RawDroneAction) else action
        output_steps.append(
            {
                "step": step_number,
                "observation": observation.model_dump(mode="json"),
                "action": action_payload,
                "next_observation": next_observation.model_dump(mode="json"),
                "reward": reward,
                "reward_breakdown": next_observation.reward_breakdown.model_dump(mode="json"),
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

    score = float(getattr(observation, "reward", 0.0) or 0.0)
    success = bool(observation.done and score > 0.0)
    print(_end_line(success, steps, score, rewards), flush=True)

    return {
        "task_id": task_id,
        "seed": args.seed,
        "policy": args.policy,
        "model": _model_name(args),
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards,
        "initial_observation": initial_observation,
        "final_observation": observation.model_dump(mode="json"),
        "trajectory": output_steps,
    }


def main() -> int:
    args = parse_args()
    runs = []

    for task_id in _task_ids(args):
        run = run_episode(args, task_id)
        runs.append(run)

        if args.output:
            write_jsonl(Path(args.output), run)

    if args.messages_output:
        messages_payload = {"runs": []}
        for run in runs:
            try:
                result = _rollout_result_from_run(run)
                messages_payload["runs"].append(
                    {
                        "task_id": run["task_id"],
                        "seed": run["seed"],
                        "policy": run["policy"],
                        "messages": trajectory_to_chat_messages(result, use_tool_calls=not args.no_tool_calls),
                    }
                )
            except Exception as exc:
                messages_payload["runs"].append(
                    {
                        "task_id": run["task_id"],
                        "seed": run["seed"],
                        "policy": run["policy"],
                        "messages": [],
                        "error": str(exc),
                    }
                )
        Path(args.messages_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.messages_output).write_text(json.dumps(messages_payload, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
