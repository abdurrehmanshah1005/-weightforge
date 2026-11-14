# Product workflow

Start the two local services using the README, or use Docker Compose.

- Overview shows actual imported model and completed-run counts.
- Model library validates supported MNIST checkpoints and preserves custom names.
- Experiment lab streams inspection, proposal, validation, candidate evaluation,
  accept/rollback and baseline events. Loss/accuracy charts distinguish temporary
  candidates from retained weights; no synthetic progress metrics are inserted.
- Saved runs survive restarts and link to reports, JSON and retained checkpoints.
- Providers & setup explains calibration, random search and optional OpenAI access.

Uploading does not start improvement automatically. Choose a model, choose a
provider and budget, then start. A proposal is an untrusted suggested scalar edit.
The calibration provider uses validation residuals; the XOR walkthrough is a
separate scripted demonstration. OpenAI requires explicit consent on every new run.

An accepted edit strictly reduces validation loss without decreasing validation
accuracy. A rollback restores previous values. Rejection or zero retained edits is
a legitimate result. Holdout is evaluated after all searches and never determines
acceptance. Repeatedly choosing runs by the reported holdout contaminates that test.

Stop is cooperative between operations. If cancellation arrives during candidate
evaluation, transaction exit restores it. A hosted request can take up to its
30-second timeout to return. Cancelled/failed runs have events but no final report.
Live sessions expire after restart; completed reports remain in Saved runs.

The frontend is in `frontend/`, independent of the Python package. Browser traffic
uses `/api/` through the frontend proxy; API keys stay in the backend environment.
For local `.env` loading, use `uv run --env-file .env` as shown in the README.
