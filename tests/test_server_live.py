"""Live-session server tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from server.app import app


def test_live_session_create_returns_visible_geometry_state():
    client = TestClient(app)

    response = client.post(
        "/live/sessions",
        json={
            "session_id": "test-live-create",
            "seed": 7,
            "scenario_family": "single_hotspot",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["session_id"] == "test-live-create"
    assert payload["observation"]["metadata"]["scenario_seed"] == 7
    assert payload["observation"]["metadata"]["scenario_family"] == "single_hotspot"
    assert payload["scene"].get("schema_version") == "rich_sim.scene.v1" or payload["scene"]["source"] == "geometry_fallback"
    assert payload["scene"]["assets"]
    assert payload["scene"]["assets"][0].get("center") or payload["scene"]["assets"][0].get("geometry")
    assert payload["event_count"] == 1


def test_live_session_state_is_retrievable_after_create():
    client = TestClient(app)
    client.post(
        "/live/sessions",
        json={"session_id": "test-live-state", "seed": 2101, "scenario_family": "single_hotspot"},
    )

    response = client.get("/live/sessions/test-live-state")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["state"]["scenario_seed"] == 2101
    assert payload["state"]["done"] is False
    assert payload["scene"]["drone"]["pose"]["z"] == 0.0
    assert payload["latest_event_id"] == 1


def test_live_session_list_reports_created_sessions():
    client = TestClient(app)
    client.post(
        "/live/sessions",
        json={"session_id": "test-live-list", "seed": 2101, "scenario_family": "single_hotspot"},
    )

    response = client.get("/live/sessions")

    assert response.status_code == 200, response.text
    sessions = response.json()["sessions"]
    assert any(session["session_id"] == "test-live-list" for session in sessions)


def test_live_session_step_records_action_result_and_geometry_scene():
    client = TestClient(app)
    client.post(
        "/live/sessions",
        json={"session_id": "test-live-step", "seed": 7, "scenario_family": "single_hotspot"},
    )

    response = client.post(
        "/live/sessions/test-live-step/step",
        json={"action": {"tool_name": "takeoff", "arguments": {"altitude_m": 18}}},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["event"]["type"] == "action-result"
    assert payload["event"]["action"]["tool_name"] == "takeoff"
    assert payload["event"]["parse_error"] is None
    assert payload["observation"]["metadata"]["step_count"] == 1
    assert payload["scene"]["drone"]["pose"]["z"] == 18.0
    assert payload["scene"]["drone"]["mode"] == "guided"
    assert "total" in payload["event"]["reward_breakdown"]


def test_live_session_event_replay_includes_start_and_result_events():
    client = TestClient(app)
    client.post(
        "/live/sessions",
        json={"session_id": "test-live-events", "seed": 7, "scenario_family": "single_hotspot"},
    )
    client.post(
        "/live/sessions/test-live-events/step",
        json={"action": {"tool_name": "get_mission_checklist", "arguments": {}}},
    )

    response = client.get("/live/sessions/test-live-events/events")

    assert response.status_code == 200, response.text
    events = response.json()["events"]
    assert [event["type"] for event in events] == [
        "session-created",
        "action-start",
        "action-result",
    ]
    assert events[1]["action"]["tool_name"] == "get_mission_checklist"
    assert events[2]["observation"]["action_result"]
    assert events[2]["scene"]["assets"]
    assert events[2]["done"] is False


def test_live_compare_runs_shared_task_policy_specs():
    client = TestClient(app)

    response = client.post(
        "/live/compare",
        json={
            "task_id": "basic_thermal_survey",
            "seed": 7,
            "max_steps": 30,
            "specs": [
                {"name": "scripted", "policy": "scripted"},
                {"name": "random", "policy": "random"},
            ],
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert [summary["name"] for summary in payload["summaries"]] == ["scripted", "random"]
    assert payload["summaries"][0]["success"] is True
    assert payload["summaries"][0]["action_sequence"][:3] == ["get_mission_checklist", "list_assets", "takeoff"]


def test_live_model_run_appends_policy_tool_call_events():
    client = TestClient(app)
    client.post(
        "/live/sessions",
        json={"session_id": "test-live-model-run", "seed": 7, "task_id": "basic_thermal_survey"},
    )

    response = client.post(
        "/live/sessions/test-live-model-run/run_model",
        json={
            "task_id": "basic_thermal_survey",
            "max_steps": 3,
            "user_instruction": "Inspect the assigned rows and start with checklist review.",
            "spec": {"name": "scripted-model", "policy": "scripted"},
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["policy"] == "scripted-model"
    assert [event["action"]["tool_name"] for event in payload["events"]] == [
        "get_mission_checklist",
        "list_assets",
        "takeoff",
    ]
    assert payload["observation"]["metadata"]["step_count"] == 3


def test_live_model_run_reuses_policy_across_incremental_requests(monkeypatch):
    client = TestClient(app)
    created = []

    class CountingPolicy:
        name = "counting"

        def __init__(self) -> None:
            self.calls = 0

        def next_action(self, observation, context):  # noqa: ANN001
            self.calls += 1
            if self.calls == 1:
                return {"tool_name": "get_mission_checklist", "arguments": {}}
            return {"tool_name": "list_assets", "arguments": {}}

    def fake_build_policy(*args, **kwargs):  # noqa: ANN002, ANN003
        policy = CountingPolicy()
        created.append(policy)
        return policy

    monkeypatch.setattr("server.live.build_policy_for_spec", fake_build_policy)
    client.post("/live/sessions", json={"session_id": "test-live-model-cache", "seed": 7})

    first = client.post(
        "/live/sessions/test-live-model-cache/run_model",
        json={"max_steps": 1, "spec": {"name": "openai-cache", "policy": "openai", "model": "gpt-5.4-mini"}},
    )
    second = client.post(
        "/live/sessions/test-live-model-cache/run_model",
        json={"max_steps": 1, "spec": {"name": "openai-cache", "policy": "openai", "model": "gpt-5.4-mini"}},
    )

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert len(created) == 1
    assert first.json()["events"][0]["action"]["tool_name"] == "get_mission_checklist"
    assert second.json()["events"][0]["action"]["tool_name"] == "list_assets"


def test_live_replay_sft_messages_into_session_events():
    client = TestClient(app)
    record = {
        "task_id": "basic_thermal_survey",
        "seed": 7,
        "messages": [
            {"role": "system", "content": "tools omitted"},
            {"role": "user", "content": "initial observation"},
            {"role": "assistant", "content": '{"tool": "get_mission_checklist", "args": {}}'},
            {"role": "user", "content": "next observation"},
            {"role": "assistant", "content": '{"tool": "takeoff", "args": {"altitude_m": 18}}'},
        ],
    }

    response = client.post(
        "/live/sessions/test-live-replay/replay",
        json={"record": record},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["session_id"] == "test-live-replay"
    assert payload["actions_replayed"] == 2
    assert payload["parse_errors"] == []
    assert [event["action"]["tool_name"] for event in payload["events"]] == ["get_mission_checklist", "takeoff"]
    assert payload["scene"]["drone"]["pose"]["z"] == 18.0


def test_live_catalog_lists_tasks_and_suites():
    client = TestClient(app)

    tasks = client.get("/live/tasks?limit=1")
    suites = client.get("/live/suites")

    assert tasks.status_code == 200, tasks.text
    assert tasks.json()["tasks"][0]["task_id"]
    assert suites.status_code == 200, suites.text
    assert any(suite["name"] == "demo" for suite in suites.json()["suites"])


def test_live_diagnostics_log_endpoint_returns_recent_entries():
    client = TestClient(app)
    client.post(
        "/live/sessions",
        json={"session_id": "test-live-diagnostics", "seed": 7, "scenario_family": "single_hotspot"},
    )

    response = client.get("/live/diagnostics/logs?limit=20")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["path"].endswith("live-server.jsonl")
    assert any(entry.get("event_type") == "session.create.response" for entry in payload["entries"])
