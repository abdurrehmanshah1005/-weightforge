"""Exercise a deployed container using only the Python standard library."""

import json
import time
import urllib.error
import urllib.request


def open_response(request):
    try:
        return urllib.request.urlopen(request, timeout=10)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise SystemExit(
            f"HTTP {exc.code} from {exc.url}: {detail}\n"
            "Expected the NEW frontend on port 8000, proxying /api to the backend. "
            "Stop any older backend on port 8000. For local development, run the "
            "backend on port 8001 and node frontend/dev-server.mjs on port 8000. "
            "For Docker, run docker compose up --build -d --wait, then inspect "
            "docker compose logs --tail 80 backend frontend."
        ) from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise SystemExit(f"Cannot reach WeightForge: {exc}. Start both services first.") from None


def get(path):
    with open_response("http://127.0.0.1:8000/api" + path) as response:
        return json.load(response)


capabilities = get("/capabilities")
assert capabilities.get("deployment") == "local-single-user", "Wrong or outdated server on port 8000"


request = urllib.request.Request(
    "http://127.0.0.1:8000/api/sessions",
    data=json.dumps({"provider": "demo", "budget": 2, "random_runs": 1}).encode(),
    headers={"Content-Type": "application/json"},
)
with open_response(request) as response:
    identity = json.load(response)["id"]
deadline = time.monotonic() + 60
while True:
    snapshot = get(f"/sessions/{identity}")
    if snapshot["status"] in {"completed", "failed", "cancelled"}:
        break
    if time.monotonic() > deadline:
        raise RuntimeError("Container experiment timed out")
    time.sleep(.25)
assert snapshot["status"] == "completed", snapshot
result = get(f"/sessions/{identity}/results")
assert get(f"/experiments/{identity}") == result
assert get("/openapi.json")["info"]["title"] == "WeightForge Lab"
with open_response(f"http://127.0.0.1:8000/api/experiments/{identity}/checkpoint") as response:
    assert len(response.read()) > 0
assert [row["accepted"] for row in result["history"]] == [True, False]
assert result["history"][0]["retained"] == result["history"][1]["retained"]
assert result["holdout"]["loss"] < result["unedited_holdout"]["loss"]
print("Container dashboard, real inference, rollback and result export passed")
