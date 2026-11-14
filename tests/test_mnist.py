import pytest
import torch
from torch import nn

from weightforge.core import TorchAdapter
from weightforge.evaluation import evaluate
from weightforge.mnist import load_checkpoint, model
from weightforge.sessions import RunConfig


def test_multiclass_metrics():
    net = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        net.weight.copy_(torch.eye(2))
    data = (torch.eye(2), torch.tensor([0, 1]))
    result = evaluate(TorchAdapter(net), data, batch_size=1)
    assert result.accuracy == 1
    assert result.loss == pytest.approx(.3132617, abs=1e-6)


def test_mnist_checkpoint_roundtrip_and_rejection(tmp_path):
    path = tmp_path / "model.pt"
    original = model()
    torch.save(original.state_dict(), path)
    restored = load_checkpoint(path)
    assert sum(p.numel() for p in restored.parameters()) == 50890
    assert torch.equal(original(torch.zeros(1, 784)), restored(torch.zeros(1, 784)))
    torch.save({"unknown": torch.zeros(2)}, path)
    with pytest.raises(ValueError):
        load_checkpoint(path)


def test_imported_model_requires_supported_provider():
    with pytest.raises(ValueError):
        RunConfig(model_id="a" * 32)
    assert RunConfig(model_id="a" * 32, provider="random").model_id == "a" * 32
    with pytest.raises(ValueError):
        RunConfig(model_id="../escape", provider="random")
