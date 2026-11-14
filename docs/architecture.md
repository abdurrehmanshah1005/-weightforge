# Architecture and invariants

## One workflow, replaceable task components

The PyTorch adapter owns weights and provides inspection plus transactional edits.
Providers only propose JSON. Evaluation decides whether validation loss strictly
decreases without reducing validation accuracy. The demo runner evaluates holdout
after search; its results never enter acceptance. The API uses the same engine.

The MVP has no database, background queue, multiuser server, training system or
automated research agent. Model loading is caller-owned; the demo constructs a
known fixture. This avoids unsafe arbitrary checkpoint uploads and unnecessary
framework infrastructure.

## Memory and complexity

Let P be scalar parameters, T parameter tensors, k coordinates in a proposal and
K retained changed coordinates. Model weights remain O(P). Adapter metadata is
O(T). Staging and rollback use O(k) scalar tensors; retained origins use O(K).
Inspection samples at most eight tensors and eight prefix values each by default,
with an 8192-byte total observation ceiling. It does not compute full-tensor means,
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

Experiment 02 inspired the additive flat-index schema, loss-plus-accuracy rule,
manual provider boundary, seeded random comparison and final-only holdout. Its
full model deepcopy, full-weight context and full initial-state comparisons were
replaced. Its old folder and checkpoints are not dependencies and remain unchanged.
