# Architecture and invariants

## One workflow, replaceable task components

The PyTorch adapter owns weights and provides inspection plus transactional edits.
Providers only propose JSON. Evaluation decides whether validation loss strictly
decreases without reducing validation accuracy. The demo runner evaluates holdout
after search; its results never enter acceptance. The API uses the same engine.

Task loading lives in tasks.py: XOR is constructed locally; MNIST loads a strictly
validated supported checkpoint and explicit evaluation splits. The preparation CLI
trains the MNIST baseline; editing uses no optimizer or backward pass. The service
has no database, external queue, multiuser accounts or automated research agent.

## Memory and complexity

Let P be scalar parameters, T parameter tensors, k coordinates in a proposal and
K retained changed coordinates. Model weights remain O(P). Adapter metadata is
O(T). Staging and rollback use O(k) scalar tensors; retained origins use O(K).
Inspection samples at most eight tensors and eight prefix values each by default,
with an 8192-byte inspection ceiling. The complete hosted-provider observation,
including diagnostics, metrics and recent decisions, is capped at 16384 bytes. It does not compute full-tensor means,
norms or serialize weights. Prefix sampling is deliberately simple and biased;
future targeted inspection should choose task-relevant components explicitly.

Validation reads only proposed coordinates and the retained origin map, not a full
initial state. It checks proposed values after destination-dtype arithmetic, so
FP16 overflow cannot sneak through finite Python deltas. Cumulative displacement
is compared to the scalar's first observed value. Returning a scalar exactly to
that value removes it from the retained map. Accepted origins survive across
proposals; rejected origins do not.

Inference is batched; data residency and forward activations still consume memory.
Sparse undo storage does not make a 7B model fit on a laptop. No whole-process RAM,
latency, large-model throughput or CUDA measurements have been established here.

## Transaction semantics

All edits are structurally/numerically validated before mutation. Unknown names,
duplicate coordinates, booleans posing as numbers, nonfinite deltas, out-of-bounds
indices, extra fields and cumulative limit violations are rejected. Rollback is
the default. Commit takes effect only on clean context exit. Exceptions after
`commit()` still roll back. Nested transactions are rejected. A reentrant lock
serializes inspection, evaluation, patch export and editing in one process.

The guarantee assumes exclusive ownership, ordinary eager modules and a forward
pass that does not mutate state. It is not crash recovery, multi-process locking,
optimizer consistency, or protection from arbitrary user code. Only touched
parameter values are restored; buffers, RNG state and arbitrary module attributes
are outside the undo log. The evaluator disables gradients and switches to eval
mode, then restores each module's previous training flag.

## Scaling boundaries

The same propose/validate/evaluate/commit workflow can be reused at larger scales,
but the current adapter rejects tied storage, quantization and noncontiguous
weights. Many transformers require a dedicated adapter. Sharding, CPU/GPU
offloading and distributed inference need ownership and synchronization designs
that this MVP does not pretend to implement. A language-model evaluator also
needs batching/tokenization, locality, efficacy, perplexity and held-out task tests.

## Live sessions and frontend

The frontend is a separate plain HTML/CSS/JavaScript service in frontend/. Nginx
serves pages and proxies /api requests to FastAPI with SSE buffering disabled. The
local Node development server provides the same route contract. FastAPI accepts
bounded run configuration and streams sequentially numbered SSE events. There is
no frontend dependency installation or build pipeline.

Each session has its own model, sparse origins, validation seed and artifact
directory. A single worker serializes experiments; at most two can be active or
queued. Up to twenty process-local session records are retained. Cooperative
cancellation is checked between operations and inside the candidate transaction.
Exceptions there trigger the existing rollback guarantee. Completed artifacts
persist; live sessions are not recovered after a process restart.

The browser receives real operation events, never executes weight edits, and
keeps candidate versus incumbent metrics separate. SSE IDs support reconnect
replay and duplicate filtering. Final holdout appears only in the terminal result.
The original manual sandbox endpoints remain isolated from dashboard experiments.

Experiment 02 inspired the additive flat-index schema, loss-plus-accuracy rule,
manual provider boundary, seeded random comparison and final-only holdout. Its
full model deepcopy, full-weight context and full initial-state comparisons were
replaced. Its old folder and checkpoints are not dependencies and remain unchanged.

## Proposal guidance and evidence

Calibration computes mean(target - probability) on validation batches and proposes
one bounded additive output-bias edit at the largest residual. This is a narrow,
task-aware direction estimate, not backpropagation or general weight repair. The
same local acceptance rule decides whether to keep it. The scripted XOR provider
is named demo and is unavailable for imported models.

OpenAI receives bounded inspection, validation metrics/residuals and three recent
decision summaries through structured JSON output. A 30-second timeout, response
size cap, explicit model configuration and consent gate bound its use. It cannot
execute code or access model files. API credentials are never included in prompts
or result artifacts. Successful response provenance is recorded; transport failures
fail the run rather than switching algorithms.

Completed results are published by atomic file replacement before the completed
event. A retained full checkpoint is written once at final export, not copied per
edit. Saved-report routes work after restart; in-flight jobs are not resumed.


Guided calibration now halves a rejected coordinate's step cap and ranks the
remaining bounded output-bias corrections. Each run starts with fresh caps. This
uses validation decisions only; it does not consult holdout or guarantee generalization.
