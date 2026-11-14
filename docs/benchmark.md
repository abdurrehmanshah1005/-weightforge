# Repeatable MNIST benchmark

The first two development runs improved validation but did not improve holdout.
The calibration method is now fixed to output residuals with rejection backoff.
This benchmark tests it across separately initialized and shuffled training runs.
It does not tune settings, stop on a favorable holdout result or select a winner.

From the repository in PowerShell:

```powershell
uv sync --locked
uv run --locked ruff check .
uv run --locked python -m pytest -q
uv run --locked python -m weightforge.benchmark --download --training-seeds 101 102 103 --epochs 3 --budget 12 --random-runs 3 --out runs/benchmark-v1
```

No backend, frontend, Docker service or API key is needed. The command trains and
evaluates sequentially on CPU with two training threads. It downloads missing
MNIST files only with `--download`. Use a new output folder for every benchmark;
existing folders are refused, including incomplete runs. It does not automatically
resume interrupted training. Do not change seeds or budgets after inspecting results.

The fixed splits use zero-based half-open ranges:

| Purpose | Dataset | Range |
| --- | --- | --- |
| Train | Official training images | [0, 12000) |
| Editing validation | Official training images | [12000, 13000) |
| Benchmark holdout | Official test images | [2000, 4000) |

The ordinary dashboard still uses test [0, 2000). We already examined that subset
while developing the algorithm, so the benchmark defaults to the next 2,000 test
images. These are fresh only if you have not previously used them to tune your
models. Future changes based on this benchmark require another untouched final
evaluation set. `--holdout-start` exists for a predeclared alternative, not for
searching for a favorable subset. Every checkpoint uses the same splits.

Training seeds 101, 102 and 103 control initialization and training shuffle. Search
seed 42 fixes each guided run's configuration; the residual proposer itself is
deterministic. Random comparisons use seeds 142, 143 and 144 on every checkpoint.
Each comparison starts from that checkpoint's unedited weights and has the same
proposal budget and edit limits. Calibration and random search still differ in
coordinate access and prior task information. This is a strategy comparison, not
a controlled proof of LLM or generic editing superiority.

Artifacts in the output folder:

- `protocol.json`: seeds, limits exposed by CLI, split ranges and source checksums;
  written before training. Per-run results also include all edit limits.
- `dataset.json`: SHA-256 of the four source dataset files.
- `checkpoints/seed-101.pt` (and other seeds): original trained models.
- `seed-101/results.json`: full guided history, all random histories, metrics,
  training seed, checkpoint SHA and runtime versions. The same folder contains
  observations, events and the final retained `model.pt`.
- `comparisons.csv`: before/after validation and holdout metrics for every trial.
- `summary.json`: all rows and checkpoint-level aggregates. Written only after
  all checkpoints finish. A failed run leaves partial artifacts, not a success summary.

Read `summary.json` first. Negative `mean_holdout_loss_delta` favors editing;
positive `mean_holdout_accuracy_delta` favors editing (multiply by 100 for percentage
points). Inspect the individual checkpoint deltas too. Random trials are averaged
within each checkpoint before averaging across checkpoints; they are not counted
as separate independently trained models. No checkpoint or trial is excluded.

Three checkpoints provide a small descriptive test, not reliable statistical
evidence of general improvement. The same training images and test images are
shared across runs, so the variation measures training-seed sensitivity rather
than new-dataset generalization. Retain all outcomes, including regressions.

To train a single baseline with a chosen seed:

```powershell
uv run --locked python -m weightforge.mnist --seed 101 --epochs 3 --out runs/mnist-seed101.pt
```

Send `summary.json` for review; keep the per-seed result files for deeper inspection.
