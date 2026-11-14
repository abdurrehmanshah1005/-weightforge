# Experiment lab

The dashboard is served by the same FastAPI process as inference. No frontend
build, Node runtime, external fonts, CDN or separate server is required to use it.
Node 24 is used only for frontend development tests.

## Run locally

```powershell
uv sync --locked
uv run --locked python -m uvicorn weightforge.api:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

Open http://127.0.0.1:8000/ and click **Start experiment**. Start with guided mode,
two proposals, seed 42 and the default edit limits. The first bias decrease should
be accepted; its reversal should be rejected and rolled back. Select a history
row or click **Review first edit** to inspect the numerical before/candidate
values. Switch the chart between loss and accuracy. After the baseline trials
finish, inspect holdout and every random trial, then download the results JSON.

Guided mode uses an explicitly declared fixture-specific bias heuristic, alternating
negative and positive deltas. It is not an LLM. With a larger budget, changes may
stop being accepted as the cumulative limit or validation optimum is reached.
Random mode uses seeded random proposals; it may find no improvement. Both modes
and their baselines use one edit per proposal and the same budget, scalar access
and numerical constraints. The guided heuristic has prior fixture knowledge.

## What the interface measures

- Green is the retained model's validation performance. Amber dots are temporary
  candidates, including rejected ones. The dashed line is the unedited model.
- A rejected or invalid proposal consumes budget. The retained curve stays at its
  previous value. History explains the decision; invalid edits never get applied.
- Stage events are emitted when real operations finish, with no artificial sleeps.
  XOR can finish almost instantly. History and the review button preserve the
  steps for inspection; the interface does not manufacture a lengthy computation.
- Elapsed time includes queueing, inference, validation and run-artifact writes.
- Undo and retained-origin byte counts are measured scalar tensor payloads. They
  exclude Python objects, metadata, activations and framework allocation overhead.
  They are not a whole-process memory benchmark or a claim of 7B model support.
- Holdout is evaluated after **all** searches finish, never streamed as a live
  optimization target. Cancelled and failed runs do not expose a final result.

## Session lifecycle and persistence

`POST /sessions` creates a fresh model experiment. One worker executes at most one
run; at most two runs may be active/queued. Configuration is limited to 50 proposals,
10 baseline trials and nine changed scalars. The process retains at most 20 session
records, evicting old terminal records without deleting their saved artifacts.

`GET /sessions/{id}/events` streams SSE events. Events have sequential IDs;
`Last-Event-ID` or the `after` query parameter replays only subsequent events.
The browser remembers its latest run ID and can replay on refresh. This works
while the server still retains the session; the registry is not reloaded on restart.

`POST /sessions/{id}/stop` requests cooperative cancellation. A pending edit rolls
back if cancellation is observed during candidate evaluation. Already accepted
edits remain in that run's discarded in-memory model; other sessions are unaffected.
No work runs in the browser and no pending tensor transaction survives shutdown.

Artifacts go to `runs/web/{id}/` by default: `config.json`, bounded observation files, `events.json`
and, for completed runs, `results.json`. Set `WEIGHTFORGE_RUNS_DIR` to change the
root. Completed JSON artifacts survive restarts, but the dashboard download routes
require a retained process-local session. Open older files directly from disk.

The original `/inspect`, `/evaluate`, `/patch` and `/proposals` routes operate on
an independent manual sandbox. They do not modify dashboard runs. The CLI also
remains independent. Use one server worker; there is no multi-process coordination.

This remains a local, unauthenticated research application. MNIST, arbitrary model
uploads and hosted LLM calls are outside this milestone.

## Development checks

```powershell
uv run --locked python -m pytest -q
uv run --locked ruff check .
node --test tests/frontend.test.mjs tests/frontend-dom.test.mjs
```

The frontend tests use explicit event fixtures to check graph semantics and wiring.
They are not inference tests. Python integration tests separately exercise real
accept/rollback and final-only holdout, plus HTTP/SSE replay and result downloads.
