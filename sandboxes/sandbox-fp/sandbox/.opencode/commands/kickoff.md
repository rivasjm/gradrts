---
description: Start working on the fixed-priority assignment task
---

Start working now. Follow `AGENTS.md`.

First create `IDENTITY.md` (who you are), then rebuild and read the current
leaderboards:

```bash
python run.py --leaderboard
```

Look at the `sched`, `gap` and `mean t` columns to see what is already on the
board (including the references you are not allowed to copy).

Then design a fixed-priority assignment algorithm of your own, implement it as a
module in `algorithms/`, and measure it:

```bash
python run.py
```

If `run.py` warns that matplotlib is missing, create the sandbox virtual
environment first (`python -m venv .venv && .venv/bin/pip install -r
requirements.txt`) and run through it (`--no-plot` also skips the chart).

You want to schedule as many systems as possible (small `gap`) while staying
efficient (small `mean t`), on **both** pools (15 and 25). Work iteratively: form
a hypothesis, implement it, measure it, and keep whatever improves the score.

Keep `CONCLUSIONS.md` (at the sandbox root) up to date as you go: what you try,
what you learn, and your final result. When you stop, also write `OVERVIEW.md`
(results per pool plus the session's wall-clock, tokens and cost), as described
in `AGENTS.md`.
