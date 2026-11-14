import pytest
import torch
from torch import nn

from weightforge import mnist


def test_training_seed_repeatability_and_split_boundary(tmp_path, monkeypatch):
    def training_data(train=True):
        assert train  # Training must never request test data.
        x = torch.zeros(13000, 784, dtype=torch.uint8)
        y = torch.zeros(13000, dtype=torch.long)
        # Out-of-range labels would fail cross entropy if validation leaked into training.
        y[12000:] = 99
        return x, y

    monkeypatch.setattr(mnist, "dataset", training_data)
    monkeypatch.setattr(mnist, "model", lambda: nn.Linear(784, 10))
    states = []
    for index, seed in enumerate((101, 101, 102)):
        path = tmp_path / f"{index}.pt"
        mnist.train_checkpoint(path, seed=seed, epochs=1)
        states.append(torch.load(path, weights_only=True))
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    assert any(not torch.equal(states[0][key], states[2][key]) for key in states[0])
    with pytest.raises(FileExistsError):
        mnist.train_checkpoint(tmp_path / "0.pt")


def test_benchmark_holdout_offset(tmp_path, monkeypatch):
    from weightforge.tasks import Task

    def data(train=True):
        count = 13000 if train else 4000
        pixels = torch.zeros(count, 784, dtype=torch.uint8)
        if not train:
            pixels[2000:] = 255
        return pixels, torch.arange(count, dtype=torch.long) % 10

    monkeypatch.setattr(mnist, "dataset", data)
    task = Task("a" * 32, 42, checkpoint=tmp_path / "external.pt", holdout_start=2000)
    assert task.checkpoint == tmp_path / "external.pt"
    assert len(task.holdout[0]) == 2000
    assert torch.all(task.holdout[0] == 1)
    assert torch.equal(task.holdout[1], data(False)[1][2000:4000])
    with pytest.raises(ValueError, match="splits"):
        Task("a" * 32, 42, holdout_start=2001)
