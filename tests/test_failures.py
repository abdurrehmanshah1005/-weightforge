from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from torch import nn

from weightforge.core import Limits, TorchAdapter
from weightforge.demo import dataset, model, observation
from weightforge.evaluation import attempt


def patch(delta):
    return {"reason": "test", "edits": [{"parameter": "2.bias", "index": 0, "delta": delta}]}


def test_evaluation_exception_restores_weights():
    net = model()
    a = TorchAdapter(net)
    calls = 0

    def fail_on_candidate(module, inputs):
        nonlocal calls
        calls += 1
        if calls == 3:  # Two incumbent batches, then candidate inference.
            raise RuntimeError("simulated inference failure")

    handle = net.register_forward_pre_hook(fail_on_candidate)
    with pytest.raises(RuntimeError, match="simulated"):
        attempt(a, patch(-1.0), dataset(42, 128))
    handle.remove()
    assert a.parameters["2.bias"].item() == 0.0
    assert not a.originals
    assert not a.active


def test_serialized_concurrent_transactions():
    a = TorchAdapter(model())

    def edit(_):
        with a.transaction(patch(-0.125)) as tx:
            tx.commit()

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(edit, range(8)))
    assert a.parameters["2.bias"].item() == -1.0
    assert len(a.originals) == 1


def test_complete_observation_bound():
    a = TorchAdapter(model(), Limits(context_bytes=32))
    with pytest.raises(ValueError, match="budget"):
        observation(a)


def test_adapter_keeps_original_parameter_storage():
    net = nn.Linear(1, 1)
    a = TorchAdapter(net)
    assert a.model is net
    assert a.parameters["weight"].data_ptr() == net.weight.data_ptr()


def test_no_grad_and_training_flags_after_exception():
    net = model()
    a = TorchAdapter(net)
    data = dataset(42, 128)
    flags = []

    def check_grad(module, inputs):
        flags.append(torch.is_grad_enabled())

    hook = net.register_forward_pre_hook(check_grad)
    attempt(a, patch(-1.0), data)
    hook.remove()
    assert flags and not any(flags)
    assert all(p.grad is None for p in net.parameters())
