"""Registered tasks own preprocessing, model loading and evaluation splits."""

import os
from pathlib import Path

import torch

from . import mnist
from .demo import dataset, model


class Task:
    def __init__(self, model_id, seed):
        self.model_id = model_id
        if model_id == "xor":
            self.name = "XOR demonstration"
            self.factory = model
            self.validation = dataset(seed, 128)
            self.holdout = dataset(seed + 1, 512)
        else:
            self.name = "MNIST digit classifier"
            self.checkpoint = Path(os.environ.get("WEIGHTFORGE_MODELS_DIR", "runs/models")) / f"{model_id}.pt"
            self.factory = self.load_mnist
            train_x, train_y = mnist.dataset()
            test_x, test_y = mnist.dataset(False)
            self.validation = (train_x[12000:13000].float() / 255, train_y[12000:13000])
            self.holdout = (test_x[:2000].float() / 255, test_y[:2000])

    def load_mnist(self):
        return mnist.load_checkpoint(self.checkpoint)

    @torch.inference_mode()
    def diagnostics(self, adapter):
        """Output calibration residuals, computed on validation only; no backward."""
        x, y = self.validation
        modes = [(module, module.training) for module in adapter.model.modules()]
        adapter.model.eval()
        try:
            residual = None
            for start in range(0, len(x), 64):
                logits = adapter.model(x[start:start + 64])
                target = y[start:start + 64]
                if target.ndim == 1:
                    difference = torch.nn.functional.one_hot(target, logits.shape[1]) - logits.softmax(1)
                else:
                    difference = target - logits.sigmoid()
                total = difference.sum(0)
                residual = total if residual is None else residual + total
            bias_name = next(name for name in reversed(adapter.parameters) if name.endswith("bias"))
            return {"output_bias": bias_name, "mean_target_minus_probability": (residual / len(x)).tolist(),
                    "validation_samples": len(x), "classes": int(residual.numel())}
        finally:
            for module, training in modes:
                module.training = training
