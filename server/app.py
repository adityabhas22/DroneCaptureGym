"""FastAPI entrypoint for OpenEnv deployment."""

from __future__ import annotations

import os
from pathlib import Path

from openenv.core.env_server.http_server import create_app
from fastapi.staticfiles import StaticFiles

from dronecaptureops.core.environment import DroneCaptureOpsEnvironment
from dronecaptureops.core.models import DroneObservation, RawDroneAction
from server.live import router as live_router


def _load_dotenv(path: Path) -> None:
    """Load local env vars for API tokens without adding a runtime dependency."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv(Path(__file__).resolve().parents[1] / ".env")


app = create_app(
    DroneCaptureOpsEnvironment,
    RawDroneAction,
    DroneObservation,
    env_name="dronecaptureops-gym",
    max_concurrent_envs=int(os.getenv("DRONECAPTUREOPS_MAX_CONCURRENT_ENVS", "64") or "64"),
)

STATIC_DIR = Path(__file__).with_name("static")
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
app.include_router(live_router)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the local OpenEnv HTTP server."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
