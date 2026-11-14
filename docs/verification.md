# Verification

## CI failure correction

Run 43608690218 on afa4e41 passed the two-container build, backend health,
real inference, rollback and checkpoint/result export. Both OS test jobs stopped
at Ruff SIM117 in test_providers.py; pytest was skipped, not failed. Combined the
two context managers without disabling the rule. Six provider tests and nine
frontend tests passed locally after the correction. Full pytest and Ruff remain
unavailable in this session. The local smoke preflight returned HTTP 503 because
the frontend could not reach port 8001; its message now identifies that condition.
Disconnected screenshots were removed from the README pending connected captures.


## Guided proposals and separate frontend — 14 November 2026

Executed for this extension:

- 9 frontend state/DOM tests passed with the separate frontend paths and `/api` proxy.
- 6 provider tests passed: residual direction/clipping, explicit consent, structured
  request/provenance, context cap, response cap/refusal, and transport failure without
  fallback. HTTP was mocked; no paid OpenAI calls were made.
- 5 dependency-free session lifecycle tests passed.
- Python source compilation, JavaScript syntax checks and Compose configuration passed.
- Actual browser checks at desktop and mobile widths confirmed page rendering,
  mobile navigation, disabled inference while disconnected, and the OpenAI selection
  link. Screenshots in `docs/screenshots` show the real backend-unavailable state.

Not executed here: the full pytest/PyTorch suite, Ruff, image build, container smoke
test, real trained-MNIST calibration benchmark, or live hosted-provider run. The
project Python launcher could not launch its underlying interpreter; Ruff execution
and Docker engine access were denied by this session's permissions. These are
outstanding checks, not passes. Added integration tests cover real XOR calibration,
MNIST architecture with explicitly synthetic data, checkpoint export, and persistent
report access after restart. CI must run them before this revision is trusted.

The preceding remote CI run `43603068143` on `5372fb0` was successful when inspected.
That result applies to the preceding revision, not this extension.

Commands for the remaining local checks:

```powershell
uv sync --locked
uv run --locked ruff check .
uv run --locked python -m pytest -q
docker compose up --build -d --wait
uv run --locked python scripts/container_smoke.py
```

Stop the Node development frontend first if it occupies port 8000. Live LLM testing
requires your configured account and explicit consent; it is never part of CI.

## Dashboard and container milestone — 14 November 2026

Actually executed in this session:

- Five standard-library session lifecycle tests passed: persistence/replay,
  queue cancellation, per-run isolation, capacity/eviction and configuration checks.
- Nine Node tests passed: separate candidate/retained curves, rejected-edit
  stability, duplicate replay filtering, final-only holdout presentation, invalid
  proposals, terminal states, DOM initialization, controls/events/review/export/reset
  wiring and accessible chart metric switching. These use declared event fixtures.
- Python source compilation and JavaScript syntax checks passed.
- `docker compose config --quiet` validated the Compose configuration.
- The dashboard loaded in the in-app browser at a narrow viewport. The static
  visual QA server deliberately had no inference service; the interface displayed
  service unavailability and disabled experiment controls correctly.

Not executed successfully in this session:

- The full Python suite, new real-PyTorch session checks and HTTP/SSE integration
  tests: this session could not start the existing Python environment or access
  parts of its installed dependencies/native PyTorch libraries. A bundled Python
  runtime could run dependency-free lifecycle tests and compile the source, but
  could not load those installed libraries. The earlier 30-test result below is
  historical and is not a test result for this new milestone.
- Ruff: the installed executable was inaccessible in the current session.
- Docker build and live container smoke: Docker Desktop's engine was not running.
  Compose validation does not establish that the image builds or runs.

The new Python/API/static-asset tests and a real container accept/rollback check
are wired into GitHub Actions. Run them in the user's normal terminal or CI before
treating this milestone as fully integration-verified:

```powershell
uv run --locked ruff check .
uv run --locked python -m pytest -q
node --test tests/frontend.test.mjs tests/frontend-dom.test.mjs
docker compose up --build -d --wait
uv run --locked python scripts/container_smoke.py
```

## Initial CLI milestone (historical)

Verified on Windows, 14 November 2026, with Python 3.12.14 and PyTorch 2.15.0+cpu
using this repository's separate environment and committed uv lockfile.

| Check | Result |
| --- | --- |
| `uv sync` | Clean project environment installed successfully |
| `uv run --locked python -m pytest -q` | 30 passed in 3.42 seconds |
| `uv run --locked ruff check .` | Passed |
| `uv run --locked ruff format --check .` | 15 files already formatted |
| Fixed-proposal CLI smoke | Completed, one proposal plus five random baseline trials |
| Seeded random CLI run | Completed, 12 proposals plus five equal-budget baseline trials |

| Metric | Unedited | Fixed-proposal smoke | Seeded random search |
| --- | --- | --- | --- |
| Validation BCE | 0.388834 | 0.206557 | 0.154663 |
| Validation accuracy | 0.601563 | 1.0 | 1.0 |
| Final holdout BCE | See full run artifacts | 0.200985 | 0.151194 |
| Final holdout accuracy | See full run artifacts | 1.0 | 1.0 |

Commands used:

```powershell
uv run --locked python -m weightforge.cli --out runs/verified-smoke --budget 1 --random-runs 5 --proposal examples/proposal.json
uv run --locked python -m weightforge.cli --out runs/verified-random --budget 12 --random-runs 5
```

The generated results and observations remain locally under the ignored `runs/`
folder. Rerun with new output folder names to reproduce them. No LLM API was
called. The smoke proposal is a declared fixture-specific heuristic. Random
search results are also a pipeline check and establish no LLM advantage.

Two upstream warnings occurred: PyTorch reported optional NumPy unavailable, and
Starlette deprecated its httpx-based test client transport. Neither affected the
30 passing tests or CLI results. The implementation does not require NumPy.
Windows Application Control blocked generated Python console launchers; running
through `python -m pytest`, `python -m weightforge.cli` and `python -m uvicorn`
avoids depending on those launcher files. README and CI use those module commands.

The initial failing test found an undeclared NumPy dependency in fixture hashing;
hashing now uses the fixed fixture's JSON values. No failures remain. Linux CI is
configured but has not run on GitHub; no CUDA, trained-model benchmark, large-model
memory measurement, 7B model, or hosted/manual LLM result has been verified.
