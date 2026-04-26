"""Run comparable policy/model specs against the same episode setup.

The comparison layer stays above the rollout runner: it builds policies,
runs each one on a fresh environment with the same task/seed/scenario, and
normalizes the result into a compact per-model summary. Heavy model
dependencies remain behind the individual policy imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from dronecaptureops.agent.policies import Policy, RandomPolicy, ScriptedPolicy
from dronecaptureops.agent.rollout import RolloutResult, RolloutRunner
from dronecaptureops.core.environment import DroneCaptureOpsEnvironment


PolicyKind = Literal["scripted", "random", "openai", "anthropic", "hf", "local_hf", "vllm"]


class ModelRunSpec(BaseModel):
    """One baseline or model checkpoint to evaluate.

    For SFT/RL-style LoRA checkpoints, set `policy` to `vllm` or `local_hf`,
    `base_model` to the pretrained model id, and `adapter_path` to the
    checkpoint directory. Omit `adapter_path` for the base model.
    """

    name: str
    policy: PolicyKind
    model: str | None = None
    base_model: str | None = None
    adapter_path: str | None = None
    api_base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 1024
    max_history_steps: int = 12
    device: str = "auto"
    trust_remote_code: bool = False
    use_tool_calls: bool = True
    enable_thinking: bool = True
    engine_kwargs: dict[str, Any] = Field(default_factory=dict)
    policy_kwargs: dict[str, Any] = Field(default_factory=dict)


class ComparisonRequest(BaseModel):
    """Shared episode inputs for a model comparison run."""

    specs: list[ModelRunSpec]
    task_id: str | None = None
    seed: int | None = None
    scenario_family: str | None = None
    user_instruction: str | None = None
    max_steps: int = 40
    include_rollouts: bool = False
    include_trace_artifacts: bool = False


class ModelRunSummary(BaseModel):
    """Normalized output for one model/spec."""

    name: str
    policy: str
    model: str | None = None
    base_model: str | None = None
    adapter_path: str | None = None
    task_id: str | None = None
    seed: int | None = None
    scenario_family: str | None = None
    steps: int
    action_sequence: list[str]
    actions: list[dict[str, Any]]
    parse_errors: list[dict[str, Any]]
    final_reward: float
    reward_breakdown: dict[str, Any]
    success: bool
    captures: list[dict[str, Any]]
    warnings: list[str]
    safety_violations: list[str]
    final_report: dict[str, Any]
    trace_artifacts: dict[str, Any] | None = None
    rollout: dict[str, Any] | None = None


class ComparisonResult(BaseModel):
    """Complete comparison output."""

    request: ComparisonRequest
    summaries: list[ModelRunSummary]


def run_model_comparison(request: ComparisonRequest) -> ComparisonResult:
    """Run every spec in `request` on the same episode configuration."""

    factory = _PolicyFactory(request.specs)
    summaries: list[ModelRunSummary] = []
    for index, spec in enumerate(request.specs, start=1):
        env = DroneCaptureOpsEnvironment()
        policy = factory.build(spec, env=env, request=request, lora_id=index)
        result = RolloutRunner(env=env).run(
            policy,
            seed=request.seed,
            task_id=request.task_id,
            scenario_family=request.scenario_family,
            max_steps=request.max_steps,
        )
        summaries.append(_summarize_result(spec, result, request=request))
    return ComparisonResult(request=request, summaries=summaries)


def build_policy_for_spec(
    spec: ModelRunSpec,
    *,
    env: DroneCaptureOpsEnvironment,
    task_id: str | None = None,
    seed: int | None = None,
    scenario_family: str | None = None,
    user_instruction: str | None = None,
    lora_id: int = 1,
    sibling_specs: list[ModelRunSpec] | None = None,
) -> Policy:
    """Build one policy from the same spec shape used by comparisons."""

    request = ComparisonRequest(
        specs=sibling_specs or [spec],
        task_id=task_id,
        seed=seed,
        scenario_family=scenario_family,
        user_instruction=user_instruction,
    )
    return _PolicyFactory(request.specs).build(spec, env=env, request=request, lora_id=lora_id)


@dataclass
class _PolicyFactory:
    specs: list[ModelRunSpec]
    _vllm_engines: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def build(
        self,
        spec: ModelRunSpec,
        *,
        env: DroneCaptureOpsEnvironment,
        request: ComparisonRequest,
        lora_id: int,
    ) -> Policy:
        if spec.policy == "scripted":
            return ScriptedPolicy(name=spec.name)
        if spec.policy == "random":
            return RandomPolicy(seed=request.seed or 0, name=spec.name)
        if spec.policy == "openai":
            from dronecaptureops.agent import OpenAIChatPolicy  # type: ignore[attr-defined]

            return OpenAIChatPolicy(
                env=env,
                task_id=request.task_id,
                user_instruction=request.user_instruction,
                model=_required_model(spec),
                api_base_url=spec.api_base_url,
                api_key=spec.api_key,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
                max_history_steps=spec.max_history_steps,
                use_tool_calls=spec.use_tool_calls,
                name=spec.name,
                **spec.policy_kwargs,
            )
        if spec.policy == "anthropic":
            from dronecaptureops.agent import AnthropicMessagesPolicy  # type: ignore[attr-defined]

            return AnthropicMessagesPolicy(
                env=env,
                task_id=request.task_id,
                user_instruction=request.user_instruction,
                model=_required_model(spec),
                api_key=spec.api_key,
                temperature=spec.temperature,
                max_tokens=spec.max_tokens,
                max_history_steps=spec.max_history_steps,
                use_tool_calls=spec.use_tool_calls,
                name=spec.name,
                **spec.policy_kwargs,
            )
        if spec.policy == "hf":
            from dronecaptureops.agent import HFInferencePolicy  # type: ignore[attr-defined]

            return HFInferencePolicy(
                env=env,
                task_id=request.task_id,
                user_instruction=request.user_instruction,
                model=_required_model(spec),
                api_base_url=spec.api_base_url or "https://router.huggingface.co/v1",
                api_key=spec.api_key,
                temperature=spec.temperature,
                top_p=spec.top_p,
                max_tokens=spec.max_tokens,
                max_history_steps=spec.max_history_steps,
                use_tool_calls=spec.use_tool_calls,
                enable_thinking=spec.enable_thinking,
                name=spec.name,
                **spec.policy_kwargs,
            )
        if spec.policy == "local_hf":
            from dronecaptureops.agent import LocalHFPolicy  # type: ignore[attr-defined]

            model_id = _required_model(spec, prefer_base=True)
            return LocalHFPolicy(
                env=env,
                task_id=request.task_id,
                user_instruction=request.user_instruction,
                model=model_id,
                base_model=spec.base_model,
                adapter_path=spec.adapter_path,
                temperature=spec.temperature,
                max_new_tokens=spec.max_tokens,
                max_history_steps=spec.max_history_steps,
                device=spec.device,
                trust_remote_code=spec.trust_remote_code,
                name=spec.name,
                **spec.policy_kwargs,
            )
        if spec.policy == "vllm":
            from dronecaptureops.agent import VLLMPolicy  # type: ignore[attr-defined]

            engine = self._vllm_engine(spec)
            return VLLMPolicy(
                engine=engine,
                env=env,
                task_id=request.task_id,
                user_instruction=request.user_instruction,
                temperature=spec.temperature,
                top_p=spec.top_p,
                max_tokens=spec.max_tokens,
                max_history_steps=spec.max_history_steps,
                name=spec.name,
                lora_request=self._lora_request(spec, lora_id),
                **spec.policy_kwargs,
            )
        raise ValueError(f"unsupported policy kind: {spec.policy!r}")

    def _vllm_engine(self, spec: ModelRunSpec) -> Any:
        from dronecaptureops.agent import VLLMEngine  # type: ignore[attr-defined]

        model_id = _required_model(spec, prefer_base=True)
        enable_lora = self._vllm_needs_lora(model_id)
        engine_kwargs = dict(spec.engine_kwargs)
        key = (
            model_id,
            enable_lora,
            tuple(sorted((key, repr(value)) for key, value in engine_kwargs.items())),
        )
        if key not in self._vllm_engines:
            self._vllm_engines[key] = VLLMEngine(
                model=model_id,
                trust_remote_code=spec.trust_remote_code,
                enable_lora=enable_lora,
                **engine_kwargs,
            )
        return self._vllm_engines[key]

    def _vllm_needs_lora(self, model_id: str) -> bool:
        return any(
            spec.policy == "vllm"
            and (spec.base_model or spec.model) == model_id
            and spec.adapter_path is not None
            for spec in self.specs
        )

    def _lora_request(self, spec: ModelRunSpec, lora_id: int) -> Any | None:
        if not spec.adapter_path:
            return None
        from vllm.lora.request import LoRARequest

        return LoRARequest(spec.name, lora_id, spec.adapter_path)


def _summarize_result(
    spec: ModelRunSpec,
    result: RolloutResult,
    *,
    request: ComparisonRequest,
) -> ModelRunSummary:
    final = result.final_observation
    warnings = _all_warnings(result)
    rollout_payload = result.model_dump(mode="json") if request.include_rollouts else None
    return ModelRunSummary(
        name=spec.name,
        policy=spec.policy,
        model=spec.model,
        base_model=spec.base_model,
        adapter_path=spec.adapter_path,
        task_id=result.task_id,
        seed=result.seed,
        scenario_family=result.scenario_family,
        steps=result.steps,
        action_sequence=[str(step.action.get("tool_name", "unknown")) for step in result.trajectory],
        actions=[_action_payload(step.action) for step in result.trajectory],
        parse_errors=[
            {"step": step.step, "error": step.parse_error}
            for step in result.trajectory
            if step.parse_error is not None
        ],
        final_reward=float(result.total_reward),
        reward_breakdown=result.reward_breakdown,
        success=bool(result.success),
        captures=list(final.get("capture_log") or []),
        warnings=warnings,
        safety_violations=_safety_violations(warnings),
        final_report=_final_report(result),
        trace_artifacts=_trace_artifacts(result) if request.include_trace_artifacts else None,
        rollout=rollout_payload,
    )


def _required_model(spec: ModelRunSpec, *, prefer_base: bool = False) -> str:
    model = (spec.base_model if prefer_base else spec.model) or spec.model or spec.base_model
    if not model:
        raise ValueError(f"{spec.policy} spec {spec.name!r} requires model or base_model")
    return model


def _action_payload(action: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_name": action.get("tool_name") or action.get("tool") or "unknown",
        "arguments": action.get("arguments") or action.get("args") or {},
    }


def _all_warnings(result: RolloutResult) -> list[str]:
    warnings: list[str] = []
    for step in result.trajectory:
        warnings.extend(str(warning) for warning in step.warnings)
        action_warnings = step.action_result.get("warnings") if isinstance(step.action_result, dict) else None
        if isinstance(action_warnings, list):
            warnings.extend(str(warning) for warning in action_warnings)
    warnings.extend(str(warning) for warning in result.final_observation.get("warnings", []) or [])
    return _dedupe(warnings)


def _safety_violations(warnings: list[str]) -> list[str]:
    markers = (
        "violation",
        "unsafe",
        "battery_exhausted",
        "invalid_gimbal",
        "no_fly",
        "privacy_capture",
        "obstacle",
    )
    return [warning for warning in warnings if any(marker in warning for marker in markers)]


def _final_report(result: RolloutResult) -> dict[str, Any]:
    submit_steps = [
        step for step in result.trajectory
        if step.action.get("tool_name") == "submit_evidence_pack"
    ]
    if not submit_steps:
        return {
            "submitted": False,
            "checklist_status": result.final_observation.get("checklist_status", {}),
        }
    last = submit_steps[-1]
    return {
        "submitted": True,
        "action_arguments": last.action.get("arguments", {}),
        "action_result": last.action_result,
        "checklist_status": result.final_observation.get("checklist_status", {}),
    }


def _trace_artifacts(result: RolloutResult) -> dict[str, Any]:
    from dronecaptureops.evaluation.tracing import build_trace_artifacts

    return build_trace_artifacts(result.model_dump(mode="json"))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)
    return output


__all__ = [
    "build_policy_for_spec",
    "ComparisonRequest",
    "ComparisonResult",
    "ModelRunSpec",
    "ModelRunSummary",
    "run_model_comparison",
]
