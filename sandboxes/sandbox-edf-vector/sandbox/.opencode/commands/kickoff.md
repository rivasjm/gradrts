---
description: Start working on the vectorized-analysis task
---

Start working now. Follow `AGENTS.md`.

First create `IDENTITY.md` (who you are), then read the reference analysis and
study what it computes:

- `frozen/analysis.py` — the scalar holistic local-EDF analysis (your reference).
- `frozen/model.py` — the system model.

Then time the scalar reference on the benchmark:

```bash
python run.py --reference
```

Implement a **vectorized** version: one `algorithms/analyze.py` exposing
`analyze(system, deadlines)` that evaluates all `K` deadline assignments of a
`(K, n_tasks)` batch at once, reproducing the reference WCRTs. Check it with:

```bash
python run.py
```

Equivalence first (values must match the reference within tolerance), then push
the speedup. Work iteratively: form a hypothesis about what is expensive, change
the implementation, measure, and keep whatever is both correct and faster.

Keep `CONCLUSIONS.md` up to date as you go, and write `OVERVIEW.md` when you
finish (final equivalence + speedup and the session's wall-clock, tokens and
cost), as described in `AGENTS.md`.
