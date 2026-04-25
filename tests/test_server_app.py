"""OpenEnv server smoke tests (issue #7).

Confirms the FastAPI app boots and the documented endpoints respond.
"""

from fastapi.testclient import TestClient

from server.app import app


def test_server_app_metadata_responds():
    client = TestClient(app)
    response = client.get("/metadata")
    assert response.status_code == 200
    payload = response.json()
    assert payload.get("name") == "DroneCaptureOps Gym"


def test_server_app_can_reset_and_step():
    client = TestClient(app)

    reset = client.post("/reset", json={"seed": 7})
    assert reset.status_code == 200, reset.text
    body = reset.json()
    assert isinstance(body, dict)

    step = client.post(
        "/step",
        json={"action": {"tool_name": "get_mission_checklist", "arguments": {}}},
    )
    assert step.status_code == 200, step.text
