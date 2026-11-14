"""Analytical XOR fixture: no training, original checkpoint, or downloads."""

import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from .core import TorchAdapter
from .evaluation import attempt, evaluate
from .providers import RandomProvider


def model():
    net = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))
    with torch.no_grad():
        net[0].weight.copy_(torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
        net[0].bias.copy_(torch.tensor([0.0, -1.0]))
        net[2].weight.copy_(torch.tensor([[4.0, -8.0]]))
        net[2].bias.fill_(0.0)  # Deliberately weak decision threshold.
    return net


def dataset(seed, count):
    if count < 4 or count % 4:
        raise ValueError("count must be a positive multiple of four")
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]).repeat(count // 4, 1)
    x += (torch.rand(count, 2, generator=g) - 0.5) * 0.2
    y = ((x[:, 0] > 0.5) != (x[:, 1] > 0.5)).float().reshape(-1, 1)
    return x, y


def observation(adapter):
    with adapter.lock:
        return _observation(adapter)


def _observation(adapter):
    probe = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    first = next(iter(adapter.parameters.values()))
    probe = probe.to(device=first.device, dtype=first.dtype)
    with adapter.lock, torch.inference_mode():
        modes = [(m, m.training) for m in adapter.model.modules()]
        try:
            adapter.model.eval()
            probabilities = adapter.model(probe).sigmoid().flatten().cpu().tolist()
        finally:
            for module, training in modes:
                module.training = training
    result = {
        "task": "Improve XOR binary logits validation loss with sparse additive edits",
        "inspection": adapter.inspect(),
        "max_delta": adapter.limits.max_delta,
        "constraints": {
            "max_edits": adapter.limits.max_edits,
            "max_changed": adapter.limits.max_changed,
            "acceptance": "Strictly lower validation BCE, no accuracy decrease",
        },
        "probe": {
            "inputs": probe.cpu().tolist(),
            "labels": [0, 1, 1, 0],
            "probabilities": probabilities,
        },
        "proposal_example": {
            "reason": "hypothesis",
            "edits": [{"parameter": "2.bias", "index": 0, "delta": -0.5}],
        },
    }
    if len(json.dumps(result, allow_nan=False).encode()) > adapter.limits.context_bytes:
        raise ValueError("Complete observation exceeds context budget")
    return result


def search(adapter, validation, provider, budget, folder=None):
    history = []
    for step in range(budget):
        obs = observation(adapter)
        if folder:
            (folder / f"observation_{step:03}.json").write_text(
                json.dumps(obs, indent=2), encoding="utf-8"
            )
        proposal = provider.propose(obs)  # Transport/exhaustion failures stop; never hide them.
        entry = {"step": step, "proposal": proposal}
        try:
            entry.update(attempt(adapter, proposal, validation))
        except ValueError as exc:
            entry.update(accepted=False, error=str(exc))
        history.append(entry)
    return history


def run(out, budget=12, random_runs=5, seed=42, provider=None):
    if not 1 <= budget <= 1000 or not 1 <= random_runs <= 100:
        raise ValueError("budget 1..1000 and random_runs 1..100 required")
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=False)
    validation, holdout = dataset(seed, 128), dataset(seed + 1, 512)
    adapter = TorchAdapter(model())
    digest = hashlib.sha256()
    for n, p in adapter.parameters.items():
        digest.update(n.encode())
        # This fixed nine-scalar fixture needs no NumPy or full model checkpoint.
        digest.update(json.dumps(p.detach().flatten().tolist()).encode())
    before = asdict(evaluate(adapter, validation))
    history = search(adapter, validation, provider or RandomProvider(seed), budget, folder)
    baselines = []
    for trial in range(random_runs):
        other = TorchAdapter(model())  # Sequential fresh fixture; no clone of edited model.
        rows = search(other, validation, RandomProvider(seed + 100 + trial), budget)
        baselines.append(
            {
                "seed": seed + 100 + trial,
                "history": rows,
                "holdout": asdict(evaluate(other, holdout)),
            }
        )
    result = {
        "config": {
            "seed": seed,
            "budget": budget,
            "random_runs": random_runs,
            "provider": type(provider).__name__ if provider else "RandomProvider",
            "limits": asdict(adapter.limits),
        },
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "fixture_sha256": digest.hexdigest(),
        "before_validation": before,
        "after_validation": asdict(evaluate(adapter, validation)),
        "unedited_holdout": asdict(evaluate(TorchAdapter(model()), holdout)),
        "final_holdout": asdict(evaluate(adapter, holdout)),
        "history": history,
        "patch": adapter.patch(),
        "random_baselines": baselines,
    }
    (folder / "results.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result
