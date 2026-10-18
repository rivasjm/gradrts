# Real-time systems: fixed-priority assignment sandbox

You are in a self-contained lab that works with **models of real-time systems**
(distributed systems of periodic tasks with end-to-end deadlines that must be
met at run time). The only goal of the lab is: **design algorithms that assign
fixed priorities so that as many systems as possible are schedulable.** You
start from a blank slate. There is no reference implementation to copy here;
you are expected to study the analysis and invent your own approach, then
iterate on it.

The lab is **portable and self-contained**: it may be placed anywhere and there
may be several independent copies of it, each used for different experiments
and possibly run by different models. Everything this copy needs is inside its
own directory; never assume anything about sibling copies.

## Before anything else: identify yourself

Before writing any algorithm, create a file called `IDENTITY.md` at the sandbox
root (next to this file) recording who is working in *this* copy. Keep it short:

```markdown
# Identity
- Provider: <your provider>
- Model: <your model name / id>
- Started: <date>
- Focus: <one line on what this copy is exploring>
```

If you do not know your exact model name, write `unknown`. Do not put secrets
or anything private in it. If `IDENTITY.md` already exists, update it with your
identity instead of deleting what is there. Because the sandbox is copied to
many places, this file is what tells each copy (and its results) apart.

## The problem

We model a **real-time system** as a set of **flows**. Each flow is an ordered
chain of **tasks**; the tasks of all flows are mapped onto a shared set of
**processors** (the mapping is *fixed*, do not change it). Within a processor,
tasks are scheduled by **fixed priorities**.

For a task, `task.priority` is a number and **a larger number means a higher
priority**. The tasks that can interfere with a task are exactly the tasks on
its processor with a priority `>=` its own (see `frozen.model.higher_priority`).

A system is **schedulable** if every flow meets its end-to-end deadline
according to the holistic response-time analysis in `frozen/analysis.py`. That
analysis is **approximate, not exact** (it is not a perfect oracle for the real
system), but it is *the reference* for this lab: "schedulable" means exactly
"the holistic analysis says so", and the reference ceiling is defined under that
test. Read `frozen/analysis.py` and `frozen/model.py`: understanding them is the
point of the lab.

The tasks' utilization, periods and deadlines come from the fixed pool; the
only thing you control is the priority ordering.

## What you write

Everything you produce goes in `algorithms/`. A module is picked up
automatically, so just add files there. An algorithm is a callable that
receives a system and assigns a priority to every task:

```python
from algorithms import register

@register
def my_policy(system):
    for task in system.tasks:
        task.priority = ...        # larger = higher priority
```

Returning a sequence of priorities in `system.tasks` order is also accepted;
mutating `task.priority` in place is the usual style. The function name is the
label used in the report. You may import anything from `frozen` (the model, the
analysis, the pool) to develop and test ideas.

## Conclusions

As you work, keep the file `CONCLUSIONS.md` at the sandbox root up to date:
what you tried, what you learned, and what you finally produced. It is the
deliverable that travels with this copy; write it for a reader who will compare
many such copies, so be concrete about the ideas, the scores and the reasons.

## Overview

When you finish, also write `OVERVIEW.md` at the sandbox root: a short, factual
summary of the run so all copies can be compared at a glance. Include, for each
pool, the schedulable count, the `gap` and `mean_time` / `total_time` of your
final algorithm(s) -- read them from `results/<pool>/leaderboard.csv` (or
`python run.py --leaderboard`) -- plus the session's wall-clock, token usage and
cost.

Get the session figures from the OpenCode CLI (already on the `PATH`; the id is
in `$OPENCODE_SESSION_ID`):

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

`OVERVIEW.md` should be factual and brief; leave the narrative to
`CONCLUSIONS.md`.

## How to run and score

From the sandbox root (this directory):

```bash
python run.py                 # evaluate every registered algorithm on the 15 pool
python run.py --pool 25       # the larger 25-task pool
python run.py --pool both     # both pools
python run.py --workers 6     # evaluate instances over up to 6 processes
python run.py --list          # list the algorithms found in algorithms/
python run.py --systems 10 --levels 6   # quick subset while developing
python run.py --per-level     # also show schedulability per utilization level
python run.py --leaderboard   # rebuild/print the leaderboard from the records
```

`run.py` applies each algorithm to every (system, utilization) instance of a
pool and reports the number of schedulable systems **and the execution time**.
It writes one **record** per method to `results/<pool>/<method>.json` and
rebuilds the per-pool leaderboards `results/<pool>/leaderboard.csv` from the
records. As the sweep progresses it writes the raw per-instance records to
`results/<pool>/raw.json` and refreshes the leaderboard and the figure
`pool.png` (schedulable systems on top, mean time below) at every utilization
level (`--no-raw`,
`--no-plot` to skip; `--workers N` to parallelize, **use at most 6**).

Each pool has a **reference** -- the row to compare against -- stored like any
other result:

- **15 tasks**: `brute-force`, an exhaustive search over every priority ordering
  (`5!^3`), run once offline. It is the **exact ceiling**. **Do not try to
  replicate brute force**: it is far too slow to be a real algorithm (its
  recorded runtime is orders of magnitude above any real method's). The `gap`
  column is how far a method is from it.
- **25 tasks**: `gdpa` is the reference, because an exhaustive search is
  intractable there (`5!^5`). It is a strong heuristic, **not an optimum**: it
  can be beaten, and beating it (negative `gap`) is a win.

`results/` also holds strong reference methods (`gdpa`, `hospa`). Their
implementations are **not** in the sandbox -- only their scores are -- and they
are there purely as targets: reach or beat them with your own efficient
algorithm.

- **Pools**: 50 systems each, swept over 20 utilization levels in [0.5, 0.9]
  -> 1000 instances per pool. **15** = 5 flows x 3 tasks on 3 processors;
  **25** = 5 flows x 5 tasks on 5 processors. Each algorithm is judged
  independently on a fresh copy of every instance.
- **Time counts**: every algorithm is timed and the report shows `mean t`
  (seconds per instance) and `total t` (seconds per sweep). A method that is
  correct but slow is not a good method. A per-instance budget is enforced
  (default 5 s, `--timeout`); exceeding it counts as unschedulable.

## Environment

This sandbox **requires matplotlib** (used by `run.py` to draw the per-pool
figure: schedulable systems on top, mean time below). Create a local virtual
environment **inside the sandbox** and install the requirements:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

`.venv/` is git-ignored. `--no-plot` skips the chart. **You may edit
`requirements.txt` to add any dependency you need** (for example a deep-learning
framework), then install it in the venv. Do not install anything outside the
sandbox.

## The pools

There are two pools, **both objectives**: `15` (5 flows x 3 tasks x 3 processors)
and `25` (5 flows x 5 tasks x 5 processors), each 50 systems over the same
utilization sweep. `python run.py` uses `15`; add `--pool 25` or `--pool both`.

Do not hard-code per-system answers and do not tune to individual systems: the
point is an algorithm that generalizes across systems, and a good one should
scale from the small pool to the larger one.

## Ground rules

- **Only create or edit files under `algorithms/`, `tmp/`, `.venv/`,
  `requirements.txt`, `CONCLUSIONS.md` and `OVERVIEW.md`.** Do not modify
  `frozen/` (the analysis, model, pool and harness are frozen) or `run.py`.
- **Scratch files go in `tmp/`.** If you need temporary files, caches or
  throwaway outputs, create them under `tmp/` inside this directory.
- **Stay inside this sandbox.** It is self-contained and portable; it may sit
  anywhere under any directory name, and there may be other copies elsewhere.
  Do not read or import anything outside this directory, whatever the
  surrounding filesystem looks like. **Never create or write files outside it,
  and never ask for permission to write outside -- the answer is always no.**
  Use `tmp/` for anything temporary. The lab is deliberately isolated so your
  ideas are your own.
- Keep the mapping task -> processor untouched; you only choose priorities.
- If an algorithm raises, the sweep keeps going and the error is reported;
  treat reported errors as bugs to fix.

## Layout

```
<sandbox root>/
  AGENTS.md          this file
  IDENTITY.md        who is working in this copy (you create it first)
  CONCLUSIONS.md     what you tried, learned and produced (you maintain it)
  OVERVIEW.md        end-of-run summary: results + session time/tokens/cost
  run.py             evaluation CLI (read-only)
  algorithms/        YOUR workspace -- add one module per idea
  frozen/            read-only evaluation core
    model.py           Task / Flow / Processor / LinearSystem
    analysis.py        HolisticFPAnalysis (the judge)
    pool.py            deterministic 15 / 25 pools
    baselines.py       pd and random reference assignments
    harness.py         the evaluation protocol (parallel + streaming)
    results.py         result records + the derived leaderboard
    chart.py           the per-pool figure (schedulables + times, 2x1)
  results/           one record per method and pool, plus leaderboard.csv
  tmp/               scratch space for temporary files
  requirements.txt   optional dependencies for a local .venv/
  .venv/             optional local virtualenv (git-ignored)
```
