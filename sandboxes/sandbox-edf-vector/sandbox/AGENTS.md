# Real-time systems: vectorized local-EDF analysis sandbox

You are in a self-contained lab that works with **models of real-time systems**
(distributed systems of periodic tasks with end-to-end deadlines). The goal of
this lab is a **performance** one: the holistic local-EDF response-time analysis
in `frozen/analysis.py` analyses **one** deadline assignment at a time, and your
job is to rewrite it so it analyses **many assignments at once** — producing the
same worst-case response times, but far faster.

You start from a blank slate. There is no reference implementation to copy
here; study the analysis, understand what it computes, and build a batched
version. Then iterate on it using the benchmark in `run.py`.

The lab is **portable and self-contained**: it may be placed anywhere and there
may be several independent copies of it, each used for different experiments and
possibly run by different models. Everything this copy needs is inside its own
directory; never assume anything about sibling copies.

## Before anything else: identify yourself

Before writing any code, create a file called `IDENTITY.md` at the sandbox root
(next to this file) recording who is working in *this* copy. Keep it short:

```markdown
# Identity
- Provider: <your provider>
- Model: <your model name / id>
- Started: <date>
- Focus: <one line on what this copy is exploring>
```

If you do not know your exact model name, write `unknown`. Do not put secrets
or anything private in it. If `IDENTITY.md` already exists, update it with your
identity instead of deleting what is there.

## The problem

A **real-time system** is a set of **flows**. Each flow is an ordered chain of
**tasks**; the tasks of all flows are mapped onto a shared set of **processors**
(the mapping is *fixed*), and every processor uses **local EDF** (earliest
deadline first). Each task has a **local deadline** (`task.deadline`) and each
flow has a fixed **end-to-end deadline** (`flow.deadline`).

The **holistic local-EDF analysis** in `frozen/analysis.py` takes one system with
one set of local deadlines and computes, for each task, its **worst-case response
time (WCRT)**. It is *holistic*: the release jitter of a task is the WCRT of its
predecessor in the flow, so processors are coupled and the analysis iterates to a
fixed point. Interference on a processor depends on the tasks' **absolute
deadline points** (the set the code calls `psi`), not on a fixed priority order.

Right now the analysis is **scalar**: to analyse `K` different local-deadline
assignments for the same system you must run it `K` times. That is the
inefficiency you are asked to remove.

Read `frozen/analysis.py` and `frozen/model.py`: understanding exactly what the
analysis computes is the whole point.

## What you write

Your code goes in `algorithms/`. The entry point is `algorithms/analyze.py`,
which must define:

```python
def analyze(system, deadlines):
    """system     : a LinearSystem (processors in EDF)
       deadlines  : shape (K, n_tasks) — K local-deadline assignments
       returns    : shape (K, n_tasks) — the WCRTs for each assignment
                     (per task, in system.tasks order)
    """
    ...
```

`deadlines[j][i]` is the local deadline of task `i` (in `system.tasks` order)
for assignment `j`. Evaluate **all K assignments together** and return their
WCRTs, matching the scalar reference analysis value for value. Do not change the
task->processor mapping, and do not rely on side effects on `system`.

You may create additional modules in `algorithms/` (and select them with
`python run.py --module algorithms.<name>`). You may add any dependency to
`requirements.txt` and install it in a local `.venv/` (see Environment).

## How to run and score

From the sandbox root (this directory):

```bash
python run.py                    # check algorithms/analyze.py (default: 25 instances x 100 assignments)
python run.py --k 200            # more assignments per instance
python run.py --instances 10 --levels 6   # a smaller, quicker sample
python run.py --reference        # only time the scalar reference
```

`run.py` builds a deterministic sample of **instances** (systems at various
utilizations) and, for each, `K` deadline assignments, then:

1. computes the **reference** WCRTs with the scalar analysis (one call per
   assignment);
2. calls your `analyze(system, deadlines)` once with the whole `(K, n_tasks)`
   batch;
3. checks **equivalence** (your values vs the reference, within a tolerance) and
   reports the **speedup** of one batched call over the `K` scalar calls.

You want both: values that **match** the reference and the **largest speedup**.
Equivalence is the gate — a fast but wrong analysis is worthless.

The default tolerances are `rtol=1e-6`, `atol=1e-6` (`--rtol`/`--atol`); the
scalar analysis uses `limit_factor=10`. Methods used by the reference analysis
(its iterative fixed points, the `psi` sets, the busy period) are all in
`frozen/analysis.py`.

## Conclusions

Keep `CONCLUSIONS.md` at the sandbox root up to date: what you tried, what you
learned about vectorizing the analysis, and the final speedup and accuracy. It
travels with this copy, so write it for a reader comparing many such copies.

## Overview

When you finish, write `OVERVIEW.md` at the sandbox root: a short, factual
summary — the final module, the equivalence result and the speedup (from
`run.py`), plus the session's wall-clock, token usage and cost. Get the session
figures from the OpenCode CLI (the id is in `$OPENCODE_SESSION_ID`):

```bash
opencode session export "$OPENCODE_SESSION_ID" > tmp/session_export.json
python3 - <<'PY'
import json
info = json.load(open("tmp/session_export.json"))["info"]
t = info["time"]
print("wall-clock: %.0f s" % ((t["updated"] - t["created"]) / 1000))
print("tokens    :", json.dumps(info["tokens"]))
print("cost      : $%.6f" % info["cost"])
PY
```

## Environment

Create a local virtual environment **inside the sandbox** and install the
requirements (numpy is provided; add anything else you need to
`requirements.txt`):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

`.venv/` is git-ignored. **You may edit `requirements.txt` to add any dependency
you need**, then install it in the venv. Do not install anything outside the
sandbox.

## Ground rules

- **Only create or edit files under `algorithms/`, `tmp/`, `.venv/`,
  `requirements.txt`, `CONCLUSIONS.md` and `OVERVIEW.md`.** Do not modify
  `frozen/` (the reference analysis, model and harness are frozen) or `run.py`.
- **Scratch files go in `tmp/`.** If you need temporary files, caches or
  throwaway outputs, create them under `tmp/` inside this directory.
- **Stay inside this sandbox.** It is self-contained and portable; it may sit
  anywhere under any directory name, and there may be other copies elsewhere.
  Do not read or import anything outside this directory, whatever the
  surrounding filesystem looks like. **Never create or write files outside it,
  and never ask for permission to write outside -- the answer is always no.**
  Use `tmp/` for anything temporary.
- Keep the mapping task -> processor untouched; you only analyse assignments,
  you do not choose them.
- If your analysis raises on an instance, `run.py` reports the failure and keeps
  going; treat reported failures as bugs to fix. If your Python uses more than
  **6 processes/threads**, that is on you — but `run.py` itself does not
  parallelize.

## Layout

```
<sandbox root>/
  AGENTS.md          this file
  IDENTITY.md        who is working in this copy (you create it first)
  CONCLUSIONS.md     what you tried, learned and produced (you maintain it)
  OVERVIEW.md        end-of-run summary: result + session time/tokens/cost
  run.py             benchmark CLI (read-only)
  algorithms/        YOUR workspace -- the vectorized analysis
    analyze.py         the entry point (implement analyze(system, deadlines))
  frozen/            read-only reference + benchmark core
    model.py           Task / Flow / Processor / LinearSystem
    analysis.py        HolisticLocalEDFAnalysis (the scalar reference)
    pool.py            the deterministic system pool
    harness.py         the benchmark (instances, reference WCRTs, speedup)
  tmp/               scratch space for temporary files
  requirements.txt   numpy (add what you need)
  .venv/             optional local virtualenv (git-ignored)
```
