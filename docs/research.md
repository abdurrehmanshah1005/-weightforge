# Research protocol and limits

WeightForge tests whether proposed sparse edits improve a measured task. It does
not assume that an LLM understands arbitrary weight arrays. XOR improvement does
not imply transformer editing, recovery of original weights, or general reasoning
improvement. The example heuristic deliberately uses knowledge of the fixture.

The default random proposer samples one visible scalar uniformly, then a delta
uniformly in [-2, 2]. Inspection exposes prefix samples, not every scalar. Manual
proposals may contain two edits or target other valid indices; such runs are not
perfectly matched to this one-edit random baseline. Report that difference instead
of claiming superiority. A fair future experiment must align coordinate access,
edits per proposal, validation calls, cumulative limits and total search budget.

Validation contains 128 jittered XOR points; holdout contains 512 separately seeded
points. Both share the same logical corners, so holdout is a limited neighborhood
check, not a new logical task. Repeated tuning against reported holdout results
contaminates it. Predeclare seeds, budgets, damage procedure and acceptance rule;
use fresh cases for final conclusions. Invalid proposals consume budget. The demo
uses no backward calls, original checkpoint, optimizer or weight training.

Each CLI run saves proposals, validation decisions, bounded observations, patch,
fixture hash, environment, seeds and full random histories/distribution. Exact
reproducibility is expected for this seeded CPU fixture in the same environment,
not guaranteed across hardware or PyTorch releases. Manual LLM provenance and
external prompts must be recorded separately. If a provider fails, a run can be
incomplete and contain observations without a final results file.

Relevant primary research:

- [ROME: Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262)
- [MEMIT: Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229)

These describe task-specific transformer editing methods; this MVP implements
neither algorithm and inherits no capability guarantee from them. Earlier chat
research claims are not treated as established results in this repository.

Future studies should measure edit efficacy, unrelated-task preservation, multiple
seeds/checkpoints, wall-clock/API cost and process peak memory. Compare no-edit,
random-search and gradient-based task baselines before attributing gains to an LLM.

## Guided and hosted-provider extension

The web default is now output-bias calibration. The CLI default remains random.
Calibration can reach any class output bias; random search samples only exposed
prefix coordinates. Compare them as different strategies, not a tightly matched
coordinate-search experiment. The hosted provider also receives validation
diagnostics. Its prompt and response metadata are artifacts, but hosted responses
are not guaranteed reproducible. Mock HTTP tests establish protocol behavior only.

MNIST training uses the first 12,000 training images; editing validation uses the
next 1,000; final evaluation uses the first 2,000 test images. An imported checkpoint
may already have seen those examples: users must supply training provenance.
Repeated model selection on displayed holdout scores invalidates independence.
Controlled synthetic-data integration tests are plumbing checks, not benchmarks.
Measure real trained checkpoints over predeclared seeds before making improvement
claims. Arbitrary transformers and 7B parameter models remain unvalidated.
