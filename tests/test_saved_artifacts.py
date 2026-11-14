import json

from fastapi.testclient import TestClient

from weightforge.api import create_app
from weightforge.sessions import SessionManager


def test_completed_artifacts_survive_manager_restart(tmp_path):
    identity = "a" * 32
    folder = tmp_path / identity
    folder.mkdir()
    result = {"config": {"seed": 1}, "initial": {"loss": 1}, "retained": {"loss": .8}}
    (folder / "results.json").write_text(json.dumps(result))
    (folder / "model.pt").write_bytes(b"checkpoint-test-artifact")
    with TestClient(create_app(SessionManager(tmp_path))) as client:
        assert client.get(f"/experiments/{identity}").json() == result
        assert client.get(f"/experiments/{identity}/download").json() == result
        assert client.get(f"/experiments/{identity}/checkpoint").content == b"checkpoint-test-artifact"
        assert client.get("/experiments").json()[0]["id"] == identity
        assert client.get(f"/sessions/{identity}").status_code == 404
        assert client.get("/experiments/not-an-id/checkpoint").status_code == 404


def test_incomplete_checkpoint_is_not_downloadable(tmp_path):
    identity = "b" * 32
    folder = tmp_path / identity
    folder.mkdir()
    (folder / "model.pt").write_bytes(b"unfinished")
    with TestClient(create_app(SessionManager(tmp_path))) as client:
        assert client.get(f"/experiments/{identity}/checkpoint").status_code == 404
