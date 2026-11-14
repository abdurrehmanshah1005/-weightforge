import json

import pytest
import torch
from torch import nn

from weightforge.core import Limits, TorchAdapter
from weightforge.demo import dataset, model, run
from weightforge.evaluation import attempt, evaluate


def proposal(delta=-0.5, name="2.bias", index=0):
    return {"reason": "test", "edits": [{"parameter": name, "index": index, "delta": delta}]}


def snapshot(adapter):
    return {n: p.detach().clone() for n, p in adapter.parameters.items()}


def assert_same(adapter, before):
    assert all(torch.equal(p, before[n]) for n, p in adapter.parameters.items())


def test_default_and_exception_rollback():
    a = TorchAdapter(model())
    before = snapshot(a)
    with a.transaction(proposal()):
        assert a.parameters["2.bias"].item() == -0.5
    assert_same(a, before)
    with pytest.raises(RuntimeError), a.transaction(proposal()) as tx:
        tx.commit()
        raise RuntimeError("evaluation failed")
    assert_same(a, before)
    assert a.originals == {}


def test_commit_and_cumulative_limit():
    a = TorchAdapter(model(), Limits(max_changed=1, max_delta=1.0))
    with a.transaction(proposal(-0.75)) as tx:
        tx.commit()
    before = snapshot(a)
    for patch in [proposal(-0.5), proposal(0.1, "0.bias")]:
        with pytest.raises(ValueError), a.transaction(patch):
            pass
        assert_same(a, before)
    with a.transaction(proposal(0.75)) as tx:
        tx.commit()
    assert not a.originals
    assert not a.patch()["edits"]


@pytest.mark.parametrize(
    "patch",
    [
        proposal(float("nan")),
        proposal(float("inf")),
        proposal(True),
        proposal(10**1000),
        proposal(index=True),
        proposal(index=-1),
        proposal(index=100),
        proposal(name="missing"),
        proposal(0),
        proposal(3),
        {"reason": "x", "edits": []},
        {**proposal(), "extra": True},
        {"reason": "x", "edits": proposal()["edits"] * 2},
        {"reason": "x", "edits": [proposal()["edits"][0], proposal(name="missing")["edits"][0]]},
    ],
)
def test_invalid_atomic(patch):
    a = TorchAdapter(model())
    before = snapshot(a)
    with pytest.raises(ValueError), a.transaction(patch):
        pass
    assert_same(a, before)


def test_dtype_overflow():
    net = nn.Linear(1, 1, bias=False).half()
    with torch.no_grad():
        net.weight.fill_(65000)
    a = TorchAdapter(net, Limits(max_delta=65000))
    with pytest.raises(ValueError), a.transaction(proposal(65000, "weight")):
        pass
    assert torch.isfinite(net.weight).all()


def test_reject_tied_and_noncontiguous():
    net = nn.Module()
    net.a = nn.Parameter(torch.ones(2))
    net.b = net.a
    with pytest.raises(ValueError, match="Shared"):
        TorchAdapter(net)
    net = nn.Linear(2, 2)
    net.weight = nn.Parameter(net.weight.t())
    with pytest.raises(ValueError, match="contiguous"):
        TorchAdapter(net)


def test_bounded_inspection_and_sparse_storage():
    a = TorchAdapter(nn.Linear(1000, 1000), Limits(max_parameters=1, samples_per_parameter=3))
    obs = a.inspect()
    assert len(json.dumps(obs).encode()) <= a.limits.context_bytes
    assert len(obs["parameters"]) == 1
    assert len(obs["parameters"][0]["samples"]) == 3
    assert obs["truncated"]
    with a.transaction(proposal(0.1, "weight")) as tx:
        tx.commit()
    assert sum(p.numel() for p in a.originals.values()) == 1


def test_nested_rejected():
    a = TorchAdapter(model())
    with (
        a.transaction(proposal()),
        pytest.raises(RuntimeError, match="Nested"),
        a.transaction(proposal()),
    ):
        pass


def test_accept_reject_and_preserve_modes():
    a = TorchAdapter(model())
    a.model.train()
    a.model[0].eval()
    modes = [m.training for m in a.model.modules()]
    data = dataset(42, 128)
    assert attempt(a, proposal(-1.0), data)["accepted"]
    before = snapshot(a)
    assert not attempt(a, proposal(1.0), data)["accepted"]
    assert_same(a, before)
    assert [m.training for m in a.model.modules()] == modes
    assert evaluate(a, data, 7).loss == pytest.approx(evaluate(a, data).loss, abs=1e-6)


def test_repeatable_demo_and_no_overwrite(tmp_path):
    one = run(tmp_path / "a", budget=3, random_runs=2)
    two = run(tmp_path / "b", budget=3, random_runs=2)
    assert one == two
    assert len(one["random_baselines"]) == 2
    assert len(one["history"]) == 3
    with pytest.raises(FileExistsError):
        run(tmp_path / "a")
