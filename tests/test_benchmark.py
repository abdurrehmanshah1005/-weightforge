"""Dependency-free protocol, orchestration and paired-summary checks."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from weightforge.benchmark import Protocol, run_benchmark, summarize


class BenchmarkTests(unittest.TestCase):
    def test_invalid_protocols(self):
        for kwargs in ({"training_seeds": (1, 1)}, {"training_seeds": (1,)},
                       {"training_seeds": (True, 2)}, {"epochs": 0},
                       {"budget": 0}, {"random_runs": 0}, {"holdout_start": 8001}):
            with self.assertRaises(ValueError):
                Protocol(**kwargs)

    def test_random_trials_are_averaged_within_checkpoint(self):
        rows = []
        for method in ("guided", "random"):
            for seed, losses in ((1, [-.1, -.3]), (2, [.1])):
                rows.extend({"method": method, "training_seed": seed,
                             "holdout_loss_delta": loss, "holdout_accuracy_delta": 0}
                            for loss in losses)
        summary = summarize(rows)["random"]
        self.assertAlmostEqual(summary["mean_holdout_loss_delta"], -.05)
        self.assertEqual(summary["checkpoint_count"], 2)
        self.assertEqual(summary["trial_count"], 3)
        self.assertEqual(summary["checkpoints_loss_improved"], 1)

    def test_protocol_precedes_training_and_every_result_is_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            for prefix in ("train", "t10k"):
                for suffix in ("images-idx3-ubyte.gz", "labels-idx1-ubyte.gz"):
                    (data / f"{prefix}-{suffix}").write_bytes(b"test-only")
            output = root / "benchmark"
            calls = []

            def train(path, seed, epochs):
                manifest = json.loads((output / "protocol.json").read_text())
                self.assertEqual(manifest["protocol"]["training_seeds"], [101, 102])
                self.assertEqual(epochs, 3)
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(str(seed).encode())
                calls.append(seed)

            def experiment(session, checkpoint, holdout_start):
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(holdout_start, 2000)
                self.assertEqual(session.config.provider, "guided")
                self.assertEqual(session.config.budget, 12)
                # Explicit synthetic result: reporting plumbing, not measured inference.
                return {"initial": {"loss": .5, "accuracy": .8},
                        "retained": {"loss": .4, "accuracy": .81},
                        "unedited_holdout": {"loss": .6, "accuracy": .8},
                        "holdout": {"loss": .7, "accuracy": .79},
                        "history": [{"accepted": True}],
                        "random_baselines": [{"seed": 142, "history": [],
                            "validation": {"loss": .5, "accuracy": .8},
                            "holdout": {"loss": .6, "accuracy": .8}}]}

            with patch.dict(os.environ, {"WEIGHTFORGE_DATA_DIR": str(data)}):
                result = run_benchmark(Protocol(training_seeds=(101, 102), random_runs=1),
                                       output, train=train, experiment=experiment)
            self.assertEqual(calls, [101, 102])
            self.assertEqual(len(result["rows"]), 4)
            self.assertEqual(result["methods"]["guided"]["checkpoints_accuracy_worsened"], 2)
            self.assertEqual(len((output / "comparisons.csv").read_text().splitlines()), 5)
            self.assertTrue((output / "seed-102/results.json").is_file())
            with self.assertRaises(FileExistsError):
                run_benchmark(Protocol(), output, train=train, experiment=experiment)

            def fail_training(path, seed, epochs):
                raise RuntimeError("training interrupted")

            failed = root / "failed"
            with (
                patch.dict(os.environ, {"WEIGHTFORGE_DATA_DIR": str(data)}),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                run_benchmark(Protocol(), failed, train=fail_training, experiment=experiment)
            self.assertTrue((failed / "protocol.json").is_file())
            self.assertFalse((failed / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
