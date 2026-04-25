"""OpenEnv-compatible DroneCaptureOps environment."""

from __future__ import annotations

from typing import Any

from openenv.core.env_server import Environment
from openenv.core.env_server.interfaces import EnvironmentMetadata
from pydantic import ValidationError

from dronecaptureops.controllers.base import DroneController
from dronecaptureops.controllers.geometry_controller import GeometryController
from dronecaptureops.core.constants import DEFAULT_DOMAIN, DEFAULT_SEED, INVALID_ACTION_LIMIT
from dronecaptureops.core.errors import ActionValidationError, EpisodeDoneError, SafetyViolationError
from dronecaptureops.core.models import DroneObservation, DroneVisibleState, RawDroneAction
from dronecaptureops.core.state import EpisodeWorld
from dronecaptureops.generation.scenario_generator import ScenarioGenerator
from dronecaptureops.rewards.reward_aggregator import RewardAggregator
from dronecaptureops.simulation.safety import SafetyChecker
from dronecaptureops.simulation.world import mark_return_status
from dronecaptureops.tools import build_tool_registry
from dronecaptureops.tools.registry import ToolRegistry
from dronecaptureops.utils.logging import action_to_log, observation_to_log


class DroneCaptureOpsEnvironment(Environment[RawDroneAction, DroneObservation, DroneVisibleState]):
    """OpenEnv environment for high-level aerial inspection operations."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        controller: DroneController | None = None,
        scenario_generator: ScenarioGenerator | None = None,
    ) -> None:
        super().__init__()
        self._controller = controller or GeometryController()
        self._scenario_generator = scenario_generator or ScenarioGenerator()
        self._safety = SafetyChecker()
        self._tools: ToolRegistry = build_tool_registry(self._controller, self._safety)
        self._rewards = RewardAggregator()
        self._world: EpisodeWorld | None = None

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        domain: str = DEFAULT_DOMAIN,
        task: str | None = None,
        task_id: str | None = None,
        **_: Any,
    ) -> DroneObservation:
        """Create a reproducible scenario and return the first observation."""

        self._world = self._scenario_generator.build(
            seed=DEFAULT_SEED if seed is None else seed,
            domain=domain,
            episode_id=episode_id,
            task_id=task_id or task,
        )
        self._controller.reset(self._world)
        self._rewards.compute(self._world)
        observation = self._render_observation("Mission initialized")
        self._world.observation_log.append(observation_to_log(observation))
        return observation

    def step(self, action: RawDroneAction | dict[str, Any], timeout_s: float | None = None, **_: Any) -> DroneObservation:
        """Validate and execute one high-level tool call."""

        world = self._require_world()
        format_valid = True
        success = False
        message = ""
        error: str | None = None
        result: dict[str, Any] = {}

        try:
            if world.done:
                raise EpisodeDoneError("episode is already done")
            typed_action = self._coerce_action(action)
            self._tools.validate(typed_action)
            result = self._tools.execute(world, typed_action)
            success = "error" not in result
            message = f"{typed_action.tool_name} executed" if success else str(result["error"])
        except (ValidationError, ActionValidationError, EpisodeDoneError) as exc:
            format_valid = False
            world.invalid_action_count += 1
            typed_action = self._fallback_action(action)
            error = str(exc)
            message = error
            result = {"error": error}
        except SafetyViolationError as exc:
            world.invalid_action_count += 1
            typed_action = self._fallback_action(action)
            error = str(exc)
            world.safety_violations.append(error)
            message = error
            result = {"error": error}

        world.step_count += 1
        mark_return_status(world)
        self._update_terminal_status(world)
        breakdown = self._rewards.compute(world, format_valid=format_valid)
        world.action_log.append(action_to_log(typed_action, success, message))
        observation = self._render_observation(message, error=error, action_result=result)
        observation.reward = breakdown.total
        observation.done = world.done
        world.observation_log.append(observation_to_log(observation))
        return observation

    @property
    def state(self) -> DroneVisibleState:
        """Return visible state only, excluding verifier-only fields."""

        world = self._require_world()
        return DroneVisibleState(
            episode_id=world.episode_id,
            step_count=world.step_count,
            domain=world.domain,
            scenario_seed=world.scenario_seed,
            task_id=world.mission.task_id,
            task_name=world.mission.task_name,
            telemetry=world.telemetry.model_copy(deep=True),
            visible_assets=[asset.model_copy(deep=True) for asset in world.assets],
            evidence_artifacts=[artifact.model_copy(deep=True) for artifact in world.evidence_artifacts],
            checklist_status=world.checklist_status.model_copy(deep=True),
            captures_taken=len(world.capture_log),
            safety_violations=list(world.safety_violations),
            done=world.done,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        """Return OpenEnv metadata."""

        return EnvironmentMetadata(
            name="DroneCaptureOps Gym",
            description="High-level drone inspection environment for active RGB/thermal evidence capture.",
            version="0.1.0",
        )

    @property
    def debug_world(self) -> EpisodeWorld:
        """Internal state accessor for tests only."""

        return self._require_world()

    def _require_world(self) -> EpisodeWorld:
        if self._world is None:
            self.reset()
        assert self._world is not None
        return self._world

    def _coerce_action(self, action: RawDroneAction | dict[str, Any]) -> RawDroneAction:
        if isinstance(action, RawDroneAction):
            return action
        return RawDroneAction(**action)

    def _fallback_action(self, action: RawDroneAction | dict[str, Any]) -> RawDroneAction:
        if isinstance(action, RawDroneAction):
            return action
        if isinstance(action, dict):
            return RawDroneAction(tool_name=str(action.get("tool_name", "invalid")), arguments=dict(action.get("arguments", {})))
        return RawDroneAction(tool_name="invalid", arguments={})

    def _update_terminal_status(self, world: EpisodeWorld) -> None:
        if world.telemetry.battery.level_pct <= 0.0:
            world.done = True
            world.termination_reason = "battery_exhausted"
            world.safety_violations.append("battery_exhausted")
        if world.invalid_action_count >= INVALID_ACTION_LIMIT:
            world.done = True
            world.termination_reason = "invalid_action_limit"
        if world.step_count >= world.max_steps:
            world.done = True
            world.termination_reason = "max_steps"

    def _render_observation(
        self,
        message: str,
        error: str | None = None,
        action_result: dict[str, Any] | None = None,
    ) -> DroneObservation:
        world = self._require_world()
        return DroneObservation(
            done=world.done,
            reward=world.reward_breakdown.total,
            system_message=message,
            error=error,
            available_tools=self._tools.names(),
            telemetry=world.telemetry.model_copy(deep=True),
            mission=world.mission.model_copy(deep=True),
            site_map=world.visible_site_map(),
            visible_assets=[asset.model_copy(deep=True) for asset in world.assets],
            evidence_artifacts=[artifact.model_copy(deep=True) for artifact in world.evidence_artifacts],
            warnings=self._visible_warnings(world),
            state_summary=self._state_summary(world),
            last_capture=world.capture_log[-1].model_copy(deep=True) if world.capture_log else None,
            capture_log=[capture.model_copy(deep=True) for capture in world.capture_log],
            checklist_status=world.checklist_status.model_copy(deep=True),
            reward_breakdown=world.reward_breakdown.model_copy(deep=True),
            action_result=action_result or {},
            metadata={
                "episode_id": world.episode_id,
                "domain": world.domain,
                "scenario_seed": world.scenario_seed,
                "step_count": world.step_count,
                "termination_reason": world.termination_reason,
            },
        )

    def _visible_warnings(self, world: EpisodeWorld) -> list[str]:
        warnings: list[str] = []
        if world.telemetry.weather_band == "high":
            warnings.append("high wind band may reduce capture stability")
        if world.telemetry.battery.level_pct < world.mission.min_battery_at_done_pct + 10.0:
            warnings.append("battery reserve margin is low")
        warnings.extend(world.safety_violations[-3:])
        return warnings

    def _state_summary(self, world: EpisodeWorld) -> dict[str, Any]:
        return {
            "mode": world.telemetry.autopilot.mode,
            "armed": world.telemetry.autopilot.armed,
            "task_id": world.mission.task_id,
            "battery_pct": world.telemetry.battery.level_pct,
            "wind_band": world.telemetry.weather_band,
            "visible_asset_count": len(world.assets),
            "artifact_count": len(world.evidence_artifacts),
            "remaining_steps": max(world.max_steps - world.step_count, 0),
            "returned_home": world.checklist_status.returned_home,
            "landed": world.checklist_status.landed,
        }
