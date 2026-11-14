"""CPU MNIST task. Explicit dataset preparation; no downloads during inference."""

import argparse
import gzip
import os
import struct
import urllib.request
from pathlib import Path

import torch
from torch import nn


def data_root():
    return Path(os.environ.get("WEIGHTFORGE_DATA_DIR", "runs/data"))


def model():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64, 10))


def dataset(train=True):
    prefix = "train" if train else "t10k"
    root = data_root()
    with gzip.open(root / f"{prefix}-images-idx3-ubyte.gz", "rb") as source:
        magic, count, rows, cols = struct.unpack(">IIII", source.read(16))
        if magic != 2051 or (rows, cols) != (28, 28):
            raise ValueError("Invalid MNIST images")
        raw = bytearray(source.read())
    if len(raw) != count * 784:
        raise ValueError("Truncated MNIST images")
    x = torch.frombuffer(raw, dtype=torch.uint8).reshape(count, 784)
    with gzip.open(root / f"{prefix}-labels-idx1-ubyte.gz", "rb") as source:
        magic, labels = struct.unpack(">II", source.read(8))
        y = torch.tensor(list(source.read()), dtype=torch.long)
    if magic != 2049 or labels != count or len(y) != count:
        raise ValueError("Invalid MNIST labels")
    return x, y


def load_checkpoint(path):
    if Path(path).stat().st_size > 2_000_000:
        raise ValueError("MNIST checkpoint exceeds 2 MB")
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = model()
    expected = net.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("Expected the WeightForge MNIST MLP state_dict")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid tensor: {name}")
    net.load_state_dict(state, strict=True)
    return net


def download_data():
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    for prefix in ("train", "t10k"):
        for suffix in ("images-idx3-ubyte.gz", "labels-idx1-ubyte.gz"):
            name = f"{prefix}-{suffix}"
            if not (root / name).exists():
                urllib.request.urlretrieve(
                    "https://ossci-datasets.s3.amazonaws.com/mnist/" + name, root / name)


def train_checkpoint(output, seed=42, epochs=3):
    """Train only on the fixed 12K training split; never read test data."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in 0..2147483647")
    if type(epochs) is not int or not 1 <= epochs <= 20:
        raise ValueError("epochs must be in 1..20")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new checkpoint output path")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    x, y = dataset()
    if len(x) < 13000:
        raise ValueError("MNIST training data must include the reserved validation split")
    net = model()
    optimizer = torch.optim.Adam(net.parameters(), lr=.001)
    for epoch in range(epochs):
        for indices in torch.randperm(12000).split(128):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(net(x[indices].float() / 255), y[indices])
            loss.backward()
            optimizer.step()
        print(f"Seed {seed}, epoch {epoch + 1}: last training batch loss {loss.item():.4f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a concurrent writer.
    with output.open("xb") as destination:
        torch.save(net.state_dict(), destination)
    return output


def prepare():
    parser = argparse.ArgumentParser(description="Prepare and train a small real MNIST classifier")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=3, choices=range(1, 21))
    parser.add_argument("--out", default="runs/mnist.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.download:
        download_data()
    output = train_checkpoint(args.out, args.seed, args.epochs)
    print(f"Saved {output}; upload it from the Models screen")


if __name__ == "__main__":
    prepare()
