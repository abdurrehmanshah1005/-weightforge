# WeightForge

A laptop-first workspace for testing sparse changes to PyTorch model weights.
Import a supported checkpoint, choose a proposer, watch measured decisions, and
export the retained model with its evaluation report.

**Status:** single-user local research application. Guided calibration and optional
OpenAI proposals are implemented; arbitrary model uploads, transformer editing and
public multiuser production deployment are not supported. Improvements are measured,
never guaranteed. See [verification](docs/verification.md) for checks actually run.


Product screenshots will be published after a connected backend and completed run
are verified. The earlier disconnected captures are not product evidence.

## How it works

1. **Prepare and import.** Train the supported MNIST classifier or bring a compatible
   float32 state_dict. Import validates names, shapes, finite values and size.
2. **Choose a proposer.** A proposal is JSON describing one parameter, flat index,
   additive delta and reason. It is a candidate change, not an approved improvement.
3. **Inspect and propose.** Bounded parameter samples and validation-only diagnostics
   guide the proposer. Raw datasets and holdout results are not sent to the LLM.
4. **Test transactionally.** Save only touched scalar values, apply temporarily,
   evaluate, and keep the edit only if validation loss falls without accuracy falling.
   Otherwise restore the previous values. No full-model deepcopy is used for undo.
5. **Compare and export.** Run seeded random comparisons from the same initial model,
   then evaluate final holdout. Saved runs expose the report, sparse patch, full JSON
   and retained checkpoint, including after a server restart.

| Proposal source | What it does | Cost / limitation |
| --- | --- | --- |
| Guided calibration | Chooses the largest mean target-minus-probability residual and adjusts that output bias | Local; narrow calibration heuristic, no backpropagation |
| Seeded random | Samples a visible prefix coordinate and bounded delta | Local; comparison baseline |
| OpenAI | Receives bounded observations and returns a structured JSON edit | API charges, backend configuration and per-run consent required |
| XOR walkthrough | Alternates a known helpful bias edit with its reversal | Scripted demonstration only; XOR only |

The default UI choice is guided calibration. Calibration and random search have
different coordinate access and prior information; matched proposal budgets do not
make this a controlled proof of superiority. A retained validation improvement can
still worsen holdout. Both are displayed separately.

## Start on Windows (Python 3.12, Node 24)

From this repository in PowerShell:

```powershell
uv python install 3.12
uv sync --locked
uv run --locked python -m uvicorn weightforge.api:create_app --factory --host 127.0.0.1 --port 8001 --workers 1
```

In a second terminal in the same folder:

```powershell
node frontend/dev-server.mjs
```

Open [WeightForge](http://127.0.0.1:8000/). No virtual-environment activation is
needed. For a quick pipeline demonstration, choose **XOR walkthrough**, two proposals,
and one baseline trial. For useful model experiments, follow the MNIST flow below.
The backend runs on 8001 during local development; the frontend on 8000 proxies
`/api/*`. [API documentation](http://127.0.0.1:8000/api/docs) uses that same origin.

## If the backend is unavailable

The local frontend on port 8000 requires FastAPI on port 8001. Keep both terminals
running. An older FastAPI process on port 8000 belongs to the previous setup; stop
it before starting the new frontend or Docker Compose.

Check the backend directly:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
Invoke-RestMethod http://127.0.0.1:8000/api/capabilities
```

For Docker, use `docker compose ps` and
`docker compose logs --tail 80 backend frontend`. The smoke script now reports
HTTP status and response details before attempting an experiment.

## Use a real small model

```powershell
uv run --locked python -m weightforge.mnist --download --epochs 3
```

Open **Model library**, name your model and import `runs/mnist.pt`. Select **Create
experiment**, choose **Guided calibration**, start with 12 proposals and 3 random
trials, then inspect **Saved runs** for holdout results and checkpoint export.
Use a different `--out` path when training again; existing checkpoints are preserved.

Supported architecture: Flatten → Linear(784,64) → ReLU → Linear(64,10), **50,890
parameters**. Training uses 12,000 MNIST examples, editing uses a separate 1,000,
and final evaluation uses 2,000 official test examples. Uploading arbitrary `.pt`
files is not supported. A new architecture requires its loader, preprocessing,
validation/holdout contract and evaluator. See [MNIST details](docs/mnist.md).


## Optional hosted LLM proposer

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `WEIGHTFORGE_LLM_MODEL`.
Choose a model available to your account that supports Chat Completions structured
JSON outputs. No model identifier or credentials are bundled.

For local development, restart the backend with:

```powershell
uv run --locked --env-file .env python -m uvicorn weightforge.api:create_app --factory --host 127.0.0.1 --port 8001 --workers 1
```

Docker Compose reads `.env` automatically. Select **OpenAI proposals** and check the
consent box before starting. The key stays server-side. Each run makes at most one
request per proposal; context is capped at 16 KiB, responses at 64 KiB, output at
1,000 tokens, and timeout at 30 seconds. No retries, tools or silent random fallback.
The engine still validates every returned edit. Provider/model/request ID and token
usage are recorded when returned. Live paid-provider execution has not been verified.

The offline CLI bridge also remains available:

```powershell
uv run --locked python -m weightforge.cli --out runs/manual --budget 1 --random-runs 3 --proposal examples/proposal.json
```

The example is a declared heuristic, not an LLM discovery. See the provider
[structured-output protocol](https://developers.openai.com/api/docs/guides/structured-outputs).


## Docker: separate frontend and backend

```powershell
docker compose up --build -d --wait
```

Only `127.0.0.1:8000` is published. The unprivileged Nginx frontend serves static
pages and proxies API/SSE to the Python backend. Models and results persist in a
named volume. Prepare MNIST data first; `runs/data` is mounted read-only. Do not run
the local frontend and Compose on port 8000 simultaneously. See [Docker operations](docs/docker.md).

## Tests and architecture

For a fixed experiment across separately trained checkpoints, see the
[MNIST benchmark instructions](docs/benchmark.md). One command trains seeds 101,
102 and 103, runs guided editing and all random comparisons, and writes a paired
summary plus CSV. It uses test images [2000, 4000) instead of the dashboard subset
already inspected during development. No backend or API key is required.

```powershell
uv run --locked python -m weightforge.benchmark --download --training-seeds 101 102 103 --epochs 3 --budget 12 --random-runs 3 --out runs/benchmark-v1
```

```powershell
uv run --locked ruff check .
uv run --locked python -m pytest -q
node --test tests/frontend.test.mjs tests/frontend-dom.test.mjs
```

CI runs Python checks on Windows and Linux, frontend tests, a CLI smoke run and a
two-container integration test covering real inference, rollback and result export.
Current local restrictions prevented executing the new full Python/container suite;
do not interpret the previous green CI run as verification of these changes.

```text
frontend/public/       Separate multi-page HTML/CSS/JavaScript product
frontend/nginx.conf   Same-origin API proxy and unbuffered SSE
frontend/Dockerfile   Frontend service
src/weightforge/      Adapter, tasks, providers, evaluator, sessions, API, CLI
tests/                Transaction, provider, task, API and frontend checks
docs/                 Architecture, protocol, deployment and screenshots
```

See [architecture](docs/architecture.md), [research limitations](docs/research.md),
[dashboard semantics](docs/dashboard.md), and [publishing](docs/publishing.md).
The service has no authentication or durable job recovery. Public production use
requires authentication/authorization, TLS, isolated resource-limited uploads and
workers, quotas, recovery, backups and operational validation. The present adapter
does not support tied, quantized, sharded or offloaded transformer weights; no 7B
model support or general capability improvement is claimed.


Guided calibration now halves a rejected coordinate's step cap and ranks the
remaining bounded output-bias corrections. Each run starts with fresh caps. This
uses validation decisions only; it does not consult holdout or guarantee generalization.
