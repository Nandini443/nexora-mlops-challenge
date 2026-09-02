# AI-USAGE.md — AI Assistance Disclosure

This document describes exactly how AI assistance was used in building this
submission, in the spirit of full transparency.

---

## What AI was used for

| Area | Tool | What it did |
|---|---|---|
| Initial code scaffolding | Antigravity (Google DeepMind) | Generated first drafts of `scoring.py`, `train.py`, `predict.py` structure |
| Bug identification | Antigravity | Flagged duplicate `--config` args, missing `--registry`/`--out` args, module-top `load_config()` |
| `src/drift.py` | Antigravity | Generated PSI implementation from a description of the algorithm |
| Test structure | Antigravity | Generated `conftest.py` synthetic fixture and test class outlines |
| `DECISIONS.md` | Antigravity | Generated initial text from bullet-point notes; content reviewed and edited |
| `README.md` | Antigravity | Generated from the project structure and run commands |
| This document | Antigravity | Structured the disclosure table |

---

## What was NOT AI-generated

| Area | Who did it |
|---|---|
| Problem framing and discipline choice (MLOps) | Human |
| Algorithm selection (3-sigma per-gateway baseline) | Human |
| Cost model analysis (€380 vs €600) and its influence on sigma tuning | Human |
| Decision to use PSI specifically (vs KS / JS divergence) | Human |
| All config parameter choices (sigma=3.0, baseline=28d, recent=7d) | Human |
| Running and verifying the pipeline end-to-end | Human |
| Recording and narrating the 6–8 min screen recording | Human |

---

## How AI outputs were verified

- Every AI-generated code file was read line-by-line before committing.
- Bugs in `train.py` and `predict.py` were caught during this review (the AI
  did not introduce them — they pre-existed and the AI identified them).
- All tests were run locally with `pytest tests/ -v` to confirm they pass.
- `validate_submission.py predictions.csv` was run to confirm the 120-row
  schema is intact after every change.
- The drift report output was manually inspected for plausibility.

---

## Reflection

AI assistance accelerated boilerplate generation (~40% of total code).
The design decisions, algorithm choices, and cost-model reasoning are entirely
human-authored — these are the parts that constitute real engineering judgment.
