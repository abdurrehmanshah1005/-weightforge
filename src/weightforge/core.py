"""Single-owner adapter and O(k) scalar undo log. No model copies."""

import json
import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Limits:
    max_edits: int = 2
    max_changed: int = 8
    max_delta: float = 2.0
    max_parameters: int = 8
    samples_per_parameter: int = 8
    context_bytes: int = 8192

    def __post_init__(self):
        for name in (
            "max_edits",
            "max_changed",
            "max_parameters",
            "samples_per_parameter",
            "context_bytes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_delta) not in (float, int) or not 0 < self.max_delta < float("inf"):
            raise ValueError("max_delta must be finite and positive")


class TorchAdapter:
    """Owns a plain eager model; callers must not mutate it outside this adapter.

    Rejects shared storage, quantized and noncontiguous parameters rather than
    silently applying incorrect flat-index updates. Metadata is O(tensor count).
    """

    def __init__(self, model: nn.Module, limits: Limits | None = None):
        self.model = model
        self.limits = limits or Limits()
        self.parameters = dict(model.named_parameters(remove_duplicate=False))
        if not self.parameters:
            raise ValueError("Model has no parameters")
        storages = set()
        for p in self.parameters.values():
            if (
                not p.is_floating_point()
                or p.layout != torch.strided
                or not p.is_contiguous()
                or p.device.type == "meta"
            ):
                raise ValueError("Only resident contiguous floating-point parameters are supported")
            key = (str(p.device), p.untyped_storage().data_ptr())
            if key in storages:
                raise ValueError("Shared/tied parameter storage is not supported in MVP")
            storages.add(key)
        self.lock = threading.RLock()
        self.originals = {}  # Only coordinates retained since this adapter was created.
        self.active = False

    def inspect(self):
        """Sample prefixes; never scan or serialize every value for statistics."""
        with self.lock, torch.no_grad():
            result = {"parameter_tensors": len(self.parameters), "parameters": []}
            for name, p in self.parameters.items():
                if len(result["parameters"]) == self.limits.max_parameters:
                    break
                row = {
                    "name": name,
                    "shape": list(p.shape),
                    "dtype": str(p.dtype),
                    "device": str(p.device),
                    "numel": p.numel(),
                    "samples": p.view(-1)[: self.limits.samples_per_parameter].cpu().tolist(),
                }
                result["parameters"].append(row)
                if len(json.dumps(result, allow_nan=False).encode()) > self.limits.context_bytes:
                    result["parameters"].pop()
                    break
            result["truncated"] = len(result["parameters"]) < len(self.parameters)
            if len(json.dumps(result, allow_nan=False).encode()) > self.limits.context_bytes:
                raise ValueError("Context budget too small")
            return result

    def validate(self, proposal):
        if type(proposal) is not dict or set(proposal) != {"reason", "edits"}:
            raise ValueError("Expected exactly reason and edits")
        if type(proposal["reason"]) is not str or len(proposal["reason"]) > 2000:
            raise ValueError("Invalid reason")
        edits = proposal["edits"]
        if type(edits) is not list or not 1 <= len(edits) <= self.limits.max_edits:
            raise ValueError("Edit count exceeds limits")
        seen, staged = set(), []
        for e in edits:
            if type(e) is not dict or set(e) != {"parameter", "index", "delta"}:
                raise ValueError("Invalid edit keys")
            name, index, delta = e["parameter"], e["index"], e["delta"]
            if type(name) is not str or name not in self.parameters:
                raise ValueError("Unknown parameter")
            p = self.parameters[name]
            if type(index) is not int or not 0 <= index < p.numel():
                raise ValueError("Invalid index")
            if (
                type(delta) not in (int, float)
                or not 0 < abs(delta) <= self.limits.max_delta
                or not math.isfinite(delta)
            ):
                raise ValueError("Invalid delta")
            key = (name, index)
            if key in seen:
                raise ValueError("Duplicate coordinate")
            seen.add(key)
            old = p.view(-1)[index].detach().clone()
            new = old + delta  # Validate actual dtype rounding and overflow before writing.
            origin = self.originals.get(key, old)
            if (
                not torch.isfinite(new).item()
                or abs(new.double().item() - origin.double().item()) > self.limits.max_delta
            ):
                raise ValueError("Cumulative delta exceeded or nonfinite result")
            staged.append((key, old, new, origin))
        changed = set(self.originals)
        for key, _, new, origin in staged:
            if torch.equal(new, origin):
                changed.discard(key)
            else:
                changed.add(key)
        if len(changed) > self.limits.max_changed:
            raise ValueError("Cumulative sparsity exceeded")
        return staged

    @contextmanager
    def transaction(self, proposal):
        with self.lock, torch.no_grad():
            if self.active:
                raise RuntimeError("Nested transactions are unsupported")
            staged = self.validate(proposal)
            self.active = True
            tx = Transaction()
            try:
                for (name, index), _, new, _ in staged:
                    self.parameters[name].view(-1)[index].copy_(new)
                yield tx
                if tx.committed:
                    for key, _, new, origin in staged:
                        if torch.equal(new, origin):
                            self.originals.pop(key, None)
                        else:
                            self.originals[key] = origin
            except BaseException:
                tx.committed = False
                raise
            finally:
                if not tx.committed:
                    for (name, index), old, _, _ in staged:
                        self.parameters[name].view(-1)[index].copy_(old)
                self.active = False

    def patch(self):
        with self.lock:
            return {
                "reason": "Cumulative accepted edits",
                "edits": [
                    {
                        "parameter": n,
                        "index": i,
                        "delta": self.parameters[n].view(-1)[i].double().item()
                        - old.double().item(),
                    }
                    for (n, i), old in sorted(self.originals.items())
                ],
            }


class Transaction:
    def __init__(self):
        self.committed = False

    def commit(self):
        self.committed = True
