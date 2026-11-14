"""Exercise a deployed container using only the Python standard library."""

import json
import time
import urllib.request


def get(path):
    with urllib.request.urlopen("http://127.0.0.1:8000/api" + path, timeout=10) as response:
        return json.load(response)


request = urllib.request.Request(
    "http://127.0.0.1:8000/api/sessions",
    data=json.dumps({"provider": "demo", "budget": 2, "random_runs": 1}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=10) as response:
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
with urllib.request.urlopen(f"http://127.0.0.1:8000/api/experiments/{identity}/checkpoint", timeout=10) as response:
    assert len(response.read()) > 0
assert [row["accepted"] for row in result["history"]] == [True, False]
assert result["history"][0]["retained"] == result["history"][1]["retained"]
assert result["holdout"]["loss"] < result["unedited_holdout"]["loss"]
print("Container dashboard, real inference, rollback and result export passed")
