"""Smoke tests for local API verification."""

from io import StringIO
from contextlib import redirect_stdout

from fastapi.testclient import TestClient

from api import app
from inference import log_end


client = TestClient(app)


def test_root_returns_ok() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Startup Survival Simulator" in response.text


def test_reset_step_state_flow() -> None:
    reset_response = client.post("/reset", json={"seed": 42})
    assert reset_response.status_code == 200
    initial_state = reset_response.json()
    assert initial_state["cash"] == 50000.0

    step_response = client.post("/step", json={"action": "increase_marketing"})
    assert step_response.status_code == 200
    payload = step_response.json()
    assert {"state", "reward", "done", "info"} <= payload.keys()

    state_response = client.get("/state")
    assert state_response.status_code == 200
    assert state_response.json() == payload["state"]


def test_tasks_endpoint_exposes_schema() -> None:
    response = client.get("/tasks")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["tasks"]) == 3
    assert "action_schema" in payload


def test_grader_and_baseline_are_valid() -> None:
    client.post("/reset", json={"seed": 42})
    response = client.get("/grader", params={"task_name": "survival"})
    assert response.status_code == 200
    assert 0.0 < response.json()["score"] < 1.0

    # The baseline endpoint should always report one entry per published task.
    baseline_response = client.get("/baseline", params={"seed": 42})
    assert baseline_response.status_code == 200
    baseline_payload = baseline_response.json()
    assert set(baseline_payload.keys()) == {"survival", "growth", "scaling"}
    for task_result in baseline_payload.values():
        assert 0.0 < task_result["score"] < 1.0


def test_inference_log_preserves_open_interval_scores() -> None:
    stream = StringIO()
    with redirect_stdout(stream):
        log_end(success=True, steps=5, score=0.999999, rewards=[1.0, 2.0])

    output = stream.getvalue().strip()
    assert "score=0.999999" in output
    assert "score=1.000" not in output
