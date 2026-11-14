import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Metrics:
    loss: float
    accuracy: float


@torch.inference_mode()
def evaluate(adapter, data, batch_size=64):
    """Binary logits task; batch inference and preserve per-module train modes."""
    x, y = data
    if batch_size < 1 or len(x) == 0 or len(x) != len(y):
        raise ValueError("Invalid evaluation data or batch size")
    with adapter.lock:
        modes = [(m, m.training) for m in adapter.model.modules()]
        adapter.model.eval()
        loss, correct = 0.0, 0
        try:
            for start in range(0, len(x), batch_size):
                target = y[start : start + batch_size]
                logits = adapter.model(x[start : start + batch_size])
                multiclass = logits.ndim == 2 and logits.shape[1] > 1 and target.ndim == 1
                if multiclass:
                    loss += nn.functional.cross_entropy(logits, target, reduction="sum").item()
                    correct += int((logits.argmax(1) == target).sum().item())
                    continue
                if logits.shape != target.shape:
                    raise ValueError("Expected binary logits matching target shape")
                loss += nn.functional.binary_cross_entropy_with_logits(
                    logits, target, reduction="sum"
                ).item()
                correct += int(((logits >= 0) == (target >= 0.5)).sum().item())
            metrics = Metrics(loss / y.numel(), correct / y.numel())
            if not math.isfinite(metrics.loss):
                raise ValueError("Nonfinite evaluation")
            return metrics
        finally:
            for module, training in modes:
                module.training = training


def attempt(adapter, proposal, validation):
    """Only validation influences acceptance. Errors restore the incumbent."""
    with adapter.lock:
        before = evaluate(adapter, validation)
        with adapter.transaction(proposal) as tx:
            after = evaluate(adapter, validation)
            accepted = after.loss < before.loss - 1e-9 and after.accuracy >= before.accuracy
            if accepted:
                tx.commit()
        return {"accepted": accepted, "before": asdict(before), "candidate": asdict(after)}
