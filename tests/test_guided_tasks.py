import json

import pytest
import torch

from weightforge.core import TorchAdapter
from weightforge.providers import CalibrationProvider
from weightforge.sessions import RunConfig, Session, run_experiment
from weightforge.tasks import Task


def test_calibration_matches_validation_residual_and_preserves_modes():
    task = Task("xor", 42)
    adapter = TorchAdapter(task.factory())
    adapter.model.train()
    before = {name: p.detach().clone() for name, p in adapter.parameters.items()}
    observation = task.diagnostics(adapter)
    x, y = task.validation
    expected = (y - adapter.model(x).sigmoid()).mean(0).item()
    assert observation["mean_target_minus_probability"] == pytest.approx([expected])
    assert adapter.model.training
    assert all(torch.equal(p, before[name]) for name, p in adapter.parameters.items())
    assert "holdout" not in json.dumps(observation)
    proposal = CalibrationProvider().propose({"diagnostics": observation, "max_delta": 2})
    assert proposal["edits"][0]["delta"] < 0


def test_guided_run_exports_checkpoint(tmp_path):
    session = Session("a" * 32, RunConfig(provider="guided", budget=2, random_runs=1), tmp_path)
    result = run_experiment(session)
    assert result["retained"]["loss"] <= result["initial"]["loss"]
    assert (tmp_path / "model.pt").is_file()
    saved = torch.load(tmp_path / "model.pt", weights_only=True)
    assert set(saved) == set(Task("xor", 42).factory().state_dict())
    assert all("holdout" not in (tmp_path / f"observation_{i:03}.json").read_text() for i in range(2))


def test_mnist_guided_pipeline_with_controlled_data(tmp_path, monkeypatch):
    from weightforge import mnist

    identity = "c" * 32
    net = mnist.model()
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.zero_()
    torch.save(net.state_dict(), tmp_path / f"{identity}.pt")
    monkeypatch.setenv("WEIGHTFORGE_MODELS_DIR", str(tmp_path))

    def data(train=True):
        count = 13000 if train else 2000
        return torch.zeros(count, 784, dtype=torch.uint8), torch.zeros(count, dtype=torch.long)

    monkeypatch.setattr(mnist, "dataset", data)
    folder = tmp_path / "run"
    folder.mkdir()
    result = run_experiment(Session("d" * 32, RunConfig(
        model_id=identity, provider="guided", budget=2, random_runs=1), folder))
    assert result["retained"]["loss"] < result["initial"]["loss"]
    assert result["task"]["validation_samples"] == 1000
    assert result["task"]["holdout_samples"] == 2000
    assert len(result["history"]) == 2
    restored = mnist.load_checkpoint(folder / "model.pt")
    assert restored.state_dict()["3.bias"][0] > 0
