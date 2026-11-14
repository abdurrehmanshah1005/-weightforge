"""Fixed-protocol, sequential MNIST benchmark; no paid providers or model selection."""

import argparse
import csv
import hashlib
import json
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from .sessions import RunConfig, Session, run_experiment


@dataclass(frozen=True)
class Protocol:
    training_seeds: tuple = (101, 102, 103)
    epochs: int = 3
    budget: int = 12
    random_runs: int = 3
    search_seed: int = 42
    # The first 2K test examples were already examined during development.
    holdout_start: int = 2000

    def __post_init__(self):
        if (not 2 <= len(self.training_seeds) <= 10
                or any(type(s) is not int or not 0 <= s < 2**31 for s in self.training_seeds)
                or len(set(self.training_seeds)) != len(self.training_seeds)):
            raise ValueError("Provide 2..10 distinct integer training seeds in 0..2147483647")
        if type(self.epochs) is not int or not 1 <= self.epochs <= 20:
            raise ValueError("epochs must be in 1..20")
        if type(self.holdout_start) is not int or not 0 <= self.holdout_start <= 8000:
            raise ValueError("holdout_start must be in 0..8000")
        RunConfig(provider="guided", budget=self.budget, random_runs=self.random_runs,
                  seed=self.search_seed)


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def comparison(seed, method, search_seed, initial, retained, original_holdout, holdout, accepted):
    return {"training_seed": seed, "method": method, "search_seed": search_seed,
            "accepted": accepted,
            "validation_loss_before": initial["loss"], "validation_loss_after": retained["loss"],
            "validation_accuracy_before": initial["accuracy"],
            "validation_accuracy_after": retained["accuracy"],
            "holdout_loss_before": original_holdout["loss"], "holdout_loss_after": holdout["loss"],
            "holdout_accuracy_before": original_holdout["accuracy"],
            "holdout_accuracy_after": holdout["accuracy"],
            "holdout_loss_delta": holdout["loss"] - original_holdout["loss"],
            "holdout_accuracy_delta": holdout["accuracy"] - original_holdout["accuracy"]}


def summarize(rows):
    summary = {}
    for method in ("guided", "random"):
        group = [r for r in rows if r["method"] == method]
        # Average random trials within each checkpoint before averaging checkpoints.
        # Trials on one checkpoint are not independent model replications.
        seeds = sorted({r["training_seed"] for r in group})
        paired = []
        for seed in seeds:
            subset = [r for r in group if r["training_seed"] == seed]
            paired.append({"training_seed": seed,
                           "loss_delta": statistics.mean(r["holdout_loss_delta"] for r in subset),
                           "accuracy_delta": statistics.mean(r["holdout_accuracy_delta"] for r in subset)})
        summary[method] = {
            "checkpoint_count": len(seeds), "trial_count": len(group),
            "mean_holdout_loss_delta": statistics.mean(r["loss_delta"] for r in paired),
            "mean_holdout_accuracy_delta": statistics.mean(r["accuracy_delta"] for r in paired),
            "checkpoints_loss_improved": sum(r["loss_delta"] < 0 for r in paired),
            "checkpoints_accuracy_improved": sum(r["accuracy_delta"] > 0 for r in paired),
            "checkpoints_accuracy_worsened": sum(r["accuracy_delta"] < 0 for r in paired),
            "paired_checkpoint_deltas": paired}
    return summary


def run_benchmark(protocol, output, *, train=None, experiment=None, download=False):
    """Callbacks allow orchestration tests without native tensor libraries."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).parent
    manifest = {"protocol_version": 1, "protocol": asdict(protocol),
                "splits": {"train": [0, 12000], "validation": [12000, 13000],
                           "test_holdout": [protocol.holdout_start, protocol.holdout_start + 2000]},
                "source_sha256": {p.name: digest(p) for p in source_root.glob("*.py")},
                "disclosure": "Same dataset splits across independently seeded checkpoints; not independent datasets. "
                    "Calibration and prefix-random search have different coordinate access. "
                    "No tuning or selection by holdout; all runs reported."}
    save_json(output / "protocol.json", manifest)
    if download or train is None:
        from .mnist import download_data, train_checkpoint

        if download:
            download_data()
        train = train or train_checkpoint
    data = Path(os.environ.get("WEIGHTFORGE_DATA_DIR", "runs/data"))
    filenames = [f"{prefix}-{suffix}" for prefix in ("train", "t10k")
                 for suffix in ("images-idx3-ubyte.gz", "labels-idx1-ubyte.gz")]
    save_json(output / "dataset.json", {name: digest(data / name) for name in filenames})
    experiment = experiment or run_experiment
    rows = []
    for seed in protocol.training_seeds:
        print(f"Training checkpoint seed {seed}", flush=True)
        checkpoint = output / "checkpoints" / f"seed-{seed}.pt"
        train(checkpoint, seed=seed, epochs=protocol.epochs)
        checkpoint_sha = digest(checkpoint)
        identity = checkpoint_sha[:32]
        folder = output / f"seed-{seed}"
        folder.mkdir()
        config = RunConfig(model_id=identity, provider="guided", seed=protocol.search_seed,
                           budget=protocol.budget, random_runs=protocol.random_runs)
        save_json(folder / "config.json", asdict(config))
        session = Session(identity, config, folder)
        print(f"Editing checkpoint seed {seed} and running random comparisons", flush=True)
        try:
            result = experiment(session, checkpoint=checkpoint, holdout_start=protocol.holdout_start)
            result["training"] = {"seed": seed, "epochs": protocol.epochs,
                                  "checkpoint_sha256": checkpoint_sha}
            save_json(folder / "results.json", result)
        finally:
            save_json(folder / "events.json", session.events)
        rows.append(comparison(seed, "guided", protocol.search_seed, result["initial"],
                               result["retained"], result["unedited_holdout"], result["holdout"],
                               sum(r["accepted"] for r in result["history"])))
        for trial in result["random_baselines"]:
            rows.append(comparison(seed, "random", trial["seed"], result["initial"],
                                   trial["validation"], result["unedited_holdout"], trial["holdout"],
                                   sum(r["accepted"] for r in trial["history"])))
    summary = {"protocol": asdict(protocol), "methods": summarize(rows), "rows": rows,
               "interpretation": "Negative loss delta and positive accuracy delta favor editing. "
                   "Small descriptive benchmark; no significance or general model-improvement claim."}
    with (output / "comparisons.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_json(output / "summary.json", summary)
    print(f"Completed all {len(protocol.training_seeds)} checkpoints. Read {output / 'summary.json'}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[101, 102, 103])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--random-runs", type=int, default=3)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--holdout-start", type=int, default=2000)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    protocol = Protocol(tuple(args.training_seeds), args.epochs, args.budget,
                        args.random_runs, args.search_seed, args.holdout_start)
    run_benchmark(protocol, args.out, download=args.download)


if __name__ == "__main__":
    main()
