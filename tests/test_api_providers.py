import json

import pytest
from fastapi.testclient import TestClient

from weightforge.api import create_app
from weightforge.core import TorchAdapter
from weightforge.demo import model, observation
from weightforge.providers import ManualLLMProvider, RandomProvider


def test_api_accept_and_reject():
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/inspect").json()["inspection"]["parameters"]
        before = client.get("/evaluate").json()
        patch = {"reason": "test", "edits": [{"parameter": "2.bias", "index": 0, "delta": -1.0}]}
        assert client.post("/proposals", json=patch).json()["accepted"]
        assert client.get("/evaluate").json()["loss"] < before["loss"]
        retained = client.get("/patch").json()
        patch["edits"][0]["index"] = True
        assert client.post("/proposals", json=patch).status_code == 422
        assert client.get("/patch").json() == retained


def test_manual_provider_and_exhaustion(tmp_path):
    obs = observation(TorchAdapter(model()))
    proposal = RandomProvider(1).propose(obs)
    path = tmp_path / "proposal.json"
    path.write_text(json.dumps(proposal), encoding="utf-8")
    p = ManualLLMProvider([path])
    assert p.propose(obs) == proposal
    with pytest.raises(RuntimeError, match="exhausted"):
        p.propose(obs)


def test_oversized_manual_file(tmp_path):
    path = tmp_path / "large.json"
    path.write_text(" " * 17000)
    with pytest.raises(ValueError, match="16 KiB"):
        ManualLLMProvider([path]).propose({})
