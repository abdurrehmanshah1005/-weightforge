"""Session transport tests run independently of PyTorch's native libraries."""

import json
import threading
import time

import pytest
from fastapi.testclient import TestClient

from weightforge.api import create_app
from weightforge.sessions import Cancelled, RunConfig, SessionManager


def wait_done(session):
    deadline = time.monotonic() + 20
    while session.snapshot()["status"] not in {"completed", "failed", "cancelled"}:
        assert time.monotonic() < deadline
        time.sleep(.01)
    return session.snapshot()


def fake_runner(session):
    session.emit("started", initial={"loss": 1, "accuracy": .5})
    return {"config": {"seed": session.config.seed}, "holdout": {"loss": .5}}


def test_persistence_and_replay(tmp_path):
    manager = SessionManager(tmp_path, fake_runner)
    try:
        session = manager.create(RunConfig())
        assert wait_done(session)["status"] == "completed"
        result = json.loads((session.folder / "results.json").read_text())
        assert result["config"]["seed"] == 42
        events, done = session.wait_events(1)
        assert done and [r["stage"] for r in events] == ["started", "completed"]
        assert "holdout" not in events[0]
        assert events[-1]["result"] == result
    finally:
        manager.close()


def test_cancel_queue_isolation_and_capacity(tmp_path):
    gate = threading.Event()

    def runner(session):
        gate.wait(3)
        if session.cancel.is_set():
            raise Cancelled()
        return {"id": session.id}

    manager = SessionManager(tmp_path, runner)
    try:
        one = manager.create(RunConfig(seed=1))
        two = manager.create(RunConfig(seed=2))
        two.cancel.set()
        with pytest.raises(ValueError, match="already"):
            manager.create(RunConfig())
        gate.set()
        assert wait_done(one)["status"] == "completed"
        assert wait_done(two)["status"] == "cancelled"
        assert one.folder != two.folder
        assert one.result == {"id": one.id} and two.result is None
    finally:
        gate.set()
        manager.close()


def test_runner_failure_is_visible(tmp_path):
    def runner(session):
        raise RuntimeError("inference unavailable")

    manager = SessionManager(tmp_path, runner)
    try:
        session = manager.create(RunConfig())
        assert wait_done(session)["status"] == "failed"
        assert session.events[-1]["message"] == "inference unavailable"
    finally:
        manager.close()


@pytest.mark.parametrize("config", [{"budget": True}, {"budget": 51}, {"seed": -1},
                                  {"provider": "llm"}, {"max_delta": float("nan")},
                                  {"max_changed": 10}, {"random_runs": 0}])
def test_config_limits(config):
    with pytest.raises(ValueError):
        RunConfig(**config)


def test_session_api_and_sse(tmp_path):
    manager = SessionManager(tmp_path, fake_runner)
    with TestClient(create_app(manager)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/sessions", json={"seed": 7})
        assert response.status_code == 201
        identity = response.json()["id"]
        wait_done(manager.get(identity))
        assert client.get(f"/sessions/{identity}").json()["status"] == "completed"
        stream = client.get(f"/sessions/{identity}/events")
        assert "text/event-stream" in stream.headers["content-type"]
        assert "event: update" in stream.text and '"stage": "completed"' in stream.text
        replay = client.get(f"/sessions/{identity}/events", headers={"Last-Event-ID": "2"})
        assert "id: 3" in replay.text and "id: 1" not in replay.text
        assert client.get(f"/sessions/{identity}/results").json()["config"]["seed"] == 7
        assert client.get("/sessions/missing").status_code == 404
        assert client.post("/sessions", json={"budget": 10000}).status_code == 422
        assert client.get(f"/sessions/{identity}/events?after=-1").status_code == 422


def test_real_guided_run_and_final_only_holdout(tmp_path):
    # This integration check requires the real PyTorch CPU runtime.
    pytest.importorskip("torch")
    manager = SessionManager(tmp_path)
    try:
        session = manager.create(RunConfig(budget=2, random_runs=1))
        snapshot = wait_done(session)
        assert snapshot["status"] == "completed", session.events
        decisions = [e for e in session.events if e["stage"] in {"accept", "rollback"}]
        assert [e["accepted"] for e in decisions] == [True, False]
        assert decisions[1]["retained"] == decisions[0]["retained"]
        assert all("holdout" not in json.dumps(e) for e in session.events[:-1])
        assert snapshot["result"]["holdout"]["loss"] < snapshot["result"]["unedited_holdout"]["loss"]
        validates = [e for e in session.events if e["stage"] == "validate"]
        assert validates[0]["undo_bytes"] == 4
    finally:
        manager.close()
