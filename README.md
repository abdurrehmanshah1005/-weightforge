# WeightForge

Laptop-first research software for proposing, testing, and retaining sparse edits
to PyTorch model weights. The workflow is **inspect → propose → validate → apply
temporarily → evaluate → accept or rollback → record**.

This is a runnable MVP, not evidence that arbitrary LLM-generated edits improve
language models. The default proposer is seeded random search. No paid service,
API key, downloaded model, GPU, optimizer, or training is needed.

## Interactive experiment lab

```powershell
uv sync --locked
uv run --locked python -m uvicorn weightforge.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Open [WeightForge Lab](http://127.0.0.1:8000/). Choose **Guided demonstration**,
two proposals and seed 42, then start. Watch the first edit improve validation;
its reversal is rejected and the improved model is preserved. Inspect the history,
switch loss/accuracy, review individual edits and download final results.

The guided proposer is a declared fixture-specific heuristic, not an LLM. The
dashboard uses actual server events and measured metrics, with separate candidate
and retained curves, cooperative stop, seeded random baselines and final-only
holdout. XOR may finish quickly; the history preserves every decision.

Prefer Docker? Start Docker Desktop and run:

```powershell
docker compose up --build -d --wait
```

The same dashboard is available at port 8000. Results persist in a named volume.
See [dashboard usage and semantics](docs/dashboard.md) and
[Docker instructions](docs/docker.md).

## Windows / Python 3.12 quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if needed,
then open PowerShell in this repository:

```powershell
uv python install 3.12
uv sync --locked
uv run --locked python -m pytest -q
uv run --locked ruff check .
uv run --locked python -m weightforge.cli --out runs/first --budget 12 --random-runs 5
```

The project explicitly supports Python 3.12 and uses the CPU PyTorch index to
avoid an unnecessary CUDA installation on a laptop. The first sync needs internet
and downloads PyTorch. A fresh run folder is required; previous results are never
overwritten. Activation is unnecessary. All generated artifacts go under `runs/`.
Linux uses the same commands. See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

Inspect `runs/first/results.json`: before/after validation loss and accuracy,
final holdout, complete proposal history, cumulative patch, seeded random trials,
fixture hash, limits, Python version and PyTorch version. Observation files contain
bounded sampled model information, never the holdout. The demo is analytical XOR
with nine scalar parameters and a deliberately weak output threshold. It is a
pipeline fixture, not an independently damaged real-world model.

## Optional LLM proposals, without an API dependency

1. Run the demo once and give only `observation_000.json` to your chosen LLM.
2. Ask it for one JSON object with exactly `reason` and `edits`, matching the example.
3. Save its response to `proposal.json`, then run:

```powershell
uv run --locked python -m weightforge.cli --out runs/manual --budget 1 --random-runs 5 --proposal proposal.json
```

Repeat `--proposal` once per step for an offline sequence. For an interactive
experiment, use `GET /inspect`, ask the LLM for a proposal, and submit it to
`POST /proposals`; record provider name/version and prompts yourself. The manual
bridge has no automatic provenance verification. Files are limited to 16 KiB.
Invalid proposals consume a search slot; missing files, exhausted providers and
other transport failures stop the run. There is no silent random fallback.

`examples/proposal.json` is an explicit bias heuristic for a deterministic smoke
check. It is not an LLM discovery or a scientific result:

```powershell
uv run --locked python -m weightforge.cli --out runs/smoke --budget 1 --random-runs 5 --proposal examples/proposal.json
```

Implement `Provider.propose(observation)` to add a hosted or local LLM later.
Providers return data only. Local validation remains authoritative. Do not give
providers filesystem tools, hidden targets, checkpoints, or holdout results.

## Basic local API

```powershell
uv run --locked python -m uvicorn weightforge.api:create_app --factory --host 127.0.0.1 --port 8000
```

Open [interactive API docs](http://127.0.0.1:8000/docs).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Local service status |
| GET | `/inspect` | Bounded proposer observation |
| GET | `/evaluate` | Current validation metrics |
| GET | `/patch` | Cumulative accepted edits |
| POST | `/proposals` | Validate, test, and conditionally retain an edit |

```powershell
$body = Get-Content examples/proposal.json -Raw
Invoke-RestMethod http://127.0.0.1:8000/proposals -Method Post -ContentType 'application/json' -Body $body
```

These original endpoints use an independent in-memory manual sandbox. Dashboard
runs use fresh isolated models and persist artifacts under `runs/web/` (or
`WEIGHTFORGE_RUNS_DIR`). One worker executes experiments and a bounded registry
retains their replayable live events. Use one process/worker. Session identities
reset on restart; saved JSON remains on disk. The service has no authentication
and is intended for loopback development.

## Adapt another model

```python
from weightforge.core import TorchAdapter, Limits
from weightforge.evaluation import attempt

adapter = TorchAdapter(your_torch_model, Limits(max_edits=2, max_changed=8, max_delta=0.1))
context = adapter.inspect()
result = attempt(adapter, proposal, (validation_inputs, binary_targets))
```

The supplied evaluator handles binary logits and batched tensors on their existing
device. For a multiclass classifier or language model, supply a task-specific
evaluator and acceptance protocol around `adapter.transaction(proposal)`.
`tx.commit()` retains an edit; exiting without commit or raising an exception
restores edited values exactly. No `deepcopy(model)` or initial full checkpoint is
used in the edit path. Users must give the adapter exclusive ownership of weights.

Only ordinary eager, resident, contiguous floating-point parameters are supported.
Tied/shared storage, quantized, sharded, offloaded, distributed and compiled-model
editing are deliberately unsupported. Custom forward methods must not mutate
parameters or buffers: rollback covers proposed parameter coordinates only.

## Tests and project structure

`src/weightforge/` contains the adapter, transaction engine, binary evaluator,
proposer protocol, CLI/demo and API. `tests/` checks atomic validation, rollback,
exception recovery, cumulative limits, bounded inspection, sparse undo storage,
overflow, unsupported layouts, seeded run repeatability and API integration.
GitHub Actions runs lint, tests and a CLI smoke run on Windows and Linux.
Frontend state/rendering checks use Node 24:

```powershell
node --test tests/frontend.test.mjs tests/frontend-dom.test.mjs
```

A separate Linux CI job builds Docker and checks the packaged dashboard, real
inference, rollback and exported results. See the verification document for what
was actually executed locally versus what remains for CI.

See [architecture](docs/architecture.md), [research limitations](docs/research.md),
[verification](docs/verification.md), and [publishing steps](docs/publishing.md).

## Next empirical steps

1. Use fresh damage cases and several predeclared seeds on small trained models.
2. Compare manual LLM proposals against the complete random distribution under the
   same proposal count, coordinate access and numerical limits.
3. Add a small multiclass task, then benchmark 100K–1M parameter models for elapsed
   time, process peak RAM and inspection size before increasing scale.
4. Add transformer-specific adapters and evaluations only after those measurements.

No 7B support or general intelligence improvement is claimed by this MVP.
