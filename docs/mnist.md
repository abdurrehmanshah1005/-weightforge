# Real-model milestone: MNIST MLP

This milestone adds a supported real classifier architecture, checkpoint import,
multiclass evaluation, a model library and a persistent results listing. It is
local research software, not a production-ready arbitrary-model optimizer.

## Prepare your model and data

From the repository in your normal PowerShell terminal:

```powershell
uv run --locked python -m weightforge.mnist --download --epochs 3
uv run --locked python -m pytest -q
uv run --locked python -m uvicorn weightforge.api:create_app --factory --host 127.0.0.1 --port 8001
```

Start `node frontend/dev-server.mjs` in a second terminal. Open `/models`, upload `runs/mnist.pt`, and select its experiment link. The model
has 50,890 parameters: Flatten, Linear(784,64), ReLU, Linear(64,10). Inputs are
28x28 grayscale pixels divided by 255. Checkpoints must be float32 state_dicts
matching that architecture exactly; model code and arbitrary pickled modules are
not accepted. Imports are capped at 2 MB and loaded with `weights_only=True`.

The preparation command explicitly downloads standard MNIST IDX gzip files from
the PyTorch dataset mirror. Training uses the first 12,000 training examples.
Editing validation uses training indices 12,000–12,999. Final evaluation uses the
first 2,000 official test examples, after all searches finish. If supplying your
own trained checkpoint, you must keep those validation/test examples out of
training. The application cannot infer the provenance of uploaded weights.

The default command trains on CPU with two threads. It does not intentionally
damage the model. Search attempts to improve the imported incumbent and can
accept no proposals. Guided calibration uses validation output residuals; random search samples visible
prefix coordinates. Optional OpenAI proposals use bounded observations. None of
these methods establishes general repair or LLM superiority. Results report validation and final test separately.

Imported models live under `runs/models`; dataset files under `runs/data`.
Environment overrides: WEIGHTFORGE_MODELS_DIR, WEIGHTFORGE_DATA_DIR. Docker mounts
the prepared host dataset read-only and stores imported models in its persistent
run volume. Prepare the data before launching a container MNIST experiment.

## Verification and remaining work

The original CI failure was confirmed in job 132463581034: `after=-1` returned
HTTP 200 because `max(after, 0)` erased the invalid value. Both cursors are now
validated before merging. Matrix fail-fast is disabled so Windows checks can
finish even if Linux fails. The preceding container job passed on GitHub.

This session passed Python compilation and the nine existing frontend tests.
Native PyTorch, pytest and Ruff remain inaccessible under this session's Windows
permissions. New multiclass/checkpoint tests are provided but have not been run
here. No measured MNIST accuracy or improvement is claimed. Run the commands
above and inspect CI before relying on this milestone.

Production readiness still requires end-to-end MNIST benchmarking, upload resource
isolation, authentication for non-local deployments, durable task recovery,
broader adapters/evaluators and measured hardware limits. Additional architectures
require an explicit task adapter, input contract and evaluation data; a weight
file alone is insufficient to judge whether a model improved.
