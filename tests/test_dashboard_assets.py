from fastapi.testclient import TestClient

from weightforge.api import create_app
from weightforge.sessions import SessionManager


def test_backend_does_not_serve_frontend(tmp_path):
    manager = SessionManager(tmp_path, lambda session: {})
    with TestClient(create_app(manager)) as client:
        assert client.get("/").status_code == 404
        assert client.get("/static/app.js").status_code == 404
        assert client.get("/health").status_code == 200
