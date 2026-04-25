"""FastAPI entrypoint for OpenEnv deployment."""

from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import DroneObservation, RawDroneAction


app = create_app(
    DroneCaptureOpsEnvironment,
    RawDroneAction,
    DroneObservation,
    env_name="dronecaptureops-gym",
    max_concurrent_envs=int(os.getenv("DRONECAPTUREOPS_MAX_CONCURRENT_ENVS", "64") or "64"),
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the local OpenEnv HTTP server."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
