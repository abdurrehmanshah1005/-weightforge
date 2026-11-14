from fastapi.testclient import TestClient

from weightforge.api import create_app
from weightforge.sessions import SessionManager


def test_packaged_dashboard_and_assets(tmp_path):
    manager = SessionManager(tmp_path, lambda session: {})
    with TestClient(create_app(manager)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "Make every edit" in page.text
        for asset in ("app.js", "lab-state.mjs", "style.css"):
            response = client.get(f"/static/{asset}")
            assert response.status_code == 200
            assert response.text.strip()
