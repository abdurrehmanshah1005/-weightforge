# Verification

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
