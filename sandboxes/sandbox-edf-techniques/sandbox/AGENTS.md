# Real-time systems: local-EDF deadline-assignment study

You are an agent working inside a small, self-contained sandbox. It evaluates
**deadline-assignment techniques** for distributed real-time systems scheduled
by **local EDF**. Nothing needs installing and nothing should be deployed: it
runs in place.

A *system* is a set of **flows** (ordered chains of tasks) mapped onto
**processors**; every processor runs **local EDF**. Each task has a **local
deadline** `task.deadline` — the parameter a technique assigns — and each flow
has a fixed **end-to-end deadline** `flow.deadline`. A system is *schedulable*
iff every flow meets its end-to-end deadline under the holistic local-EDF
response-time analysis in `frozen/analysis.py`.

## Quick start

From the sandbox root (this directory):

```bash
python run.py                 # every registered technique: 50 systems x 20 levels = 1000 instances
python run.py --per-level     # also the schedulable count per utilization level
python run.py -q              # quiet
python run.py --systems 10 --levels 6   # fast smoke run
```

The evaluation function is `frozen/harness.py`:

```python
from frozen import harness
report = harness.evaluate()                  # every technique registered in algorithms/
report["techniques"]["pd"]["schedulable"]    # e.g. 708
report = harness.evaluate({"my": my_fn})     # or an explicit {name: technique} dict
```

### Interpreter (reproducibility)

Run everything with the repository virtualenv — `../../../.venv/bin/python`
(Python 3.13); in this session `python` and `python3` already resolve to it.
The response-time analysis is **numerically sensitive** close to the
schedulability boundary, so a different interpreter (e.g. the system `python3`,
3.9) shifts the counts by a few instances and will not match the numbers below.
To be safe, always invoke the venv explicitly in scripts and experiments.

## Layout

```
frozen/analysis.py   the judge: holistic local-EDF response-time analysis
frozen/model.py      Task / Flow / Processor / LinearSystem
frozen/pool.py       deterministic example pool (seed 42): 5 flows x 3 tasks x 3 procs
frozen/harness.py    evaluate(...): counts schedulable instances per technique
algorithms/          YOUR workspace: one module per technique
algorithms/__init__.py  the @register registry + auto-discovery
algorithms/pd.py        pd and pd_global (the shipped baselines)
run.py               the runner
```

## Registering a technique

Add a module under `algorithms/` and decorate the callable with `@register`
(the function name is the label). Every module in the package is imported
automatically — no list to edit:

```python
from algorithms import register

@register
def my_technique(system):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        for t in flow.tasks:
            t.deadline = ...      # what the local-EDF analysis judges
```

Rules: **do not modify `frozen/` or `run.py`**; keep new work in `algorithms/`;
put scratch files in `tmp/` (create it). Use `harness.evaluate({...})` for
ad-hoc experiments without registering anything.

## Isolation (work only inside this sandbox)

This sandbox is a **standalone** study. Work **only inside this directory**: do
not read, write, list or import anything from other sandboxes — in particular
the sibling `sandbox-edf-derive` — nor anywhere else on the filesystem.
Depending on external files breaks comparability and reproducibility. If you
need something from elsewhere, recreate it here.

## Step 0 — understand the holistic analysis *in detail* (do this first)

The research questions are about **why** the counts move, so a black-box sweep is
not enough: you must be able to explain every effect **from the analysis itself**.
Before running any experiment, study `frozen/analysis.py` (and `frozen/model.py`)
line by line and **write your understanding down** in `NOTES.md` (a dedicated
"Analysis" section). Work through, at least:

- **The model.** A flow is a chain of tasks mapped onto processors; a task's
  release **jitter** is the maximum WCRT of its predecessors, so processors are
  *coupled* and the analysis iterates to a fixed point. Schedulability is a
  property of the **flow's end-to-end deadline** — the task's local deadline is
  **not** a requirement, it is a free parameter. Keep this distinction explicit.
- **The interference term `w_i(t, D)` (eq 1).** Find *exactly* where the
  interfering task's local deadline enters. What does `D < task.deadline` do, and
  why does a **later** local deadline on an interferer mean **fewer** interfering
  activations? This is the crux of "larger deadlines help".
- **The `psi` sets (eq 4 and eq 6).** Release points of the interfering tasks and
  of the analysed task; which window of `psi` is considered for each activation
  index `p`.
- **The busy period (eq 5)** and the `p`-loop over the analysed task's activations.
- **The fixed point `w_ab` (eq 8)** and the response `r_ab = w_ab - psi +
  deadline + jitter` (eq 9). Here `deadline` is the *analysed* task's local
  deadline — enumerate **every** place a local deadline appears (analysed task
  vs interferer) and in which direction it pushes the response time.
- **The `limit_factor` abort.** When a provisional response exceeds
  `flow.deadline * limit_factor` the analysis aborts and, with `reset=True`, the
  system counts as **unschedulable** (`harness.is_schedulable` uses
  `limit_factor=1`). Note how this interacts with very large deadlines.

The analysis is explicitly **approximate, not exact**. Write, in your own words,
where the local deadlines enter and how that can make *larger* deadlines help.
Then, for **each** research question, state the **prediction implied by the
mechanism** *before* running the experiment, and let the numbers confirm or
refute it. A conclusion that could have been reached without reading the analysis
is not a satisfying answer here.

## The observation to explain (the task)

The only hard requirement is the flow's **end-to-end** deadline; the **local**
deadline is a *free scheduling parameter* (under EDF it also fixes the priority
order). Two shipped techniques, both proportional to WCET:

- `pd` — `task.deadline = wcet * flow.deadline / sum(wcet)` → **708 / 1000**
- `pd_global` — the same, accumulated along the flow (cumulative) → **744 / 1000**

It goes further. With `d_i = share_i + γ · (sum of previous shares)`:

| γ | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| schedulable | 708 | 773 | 804 | 813 | 813 | **815** | 803 | 786 | 773 | 758 | 744 |

And scaling **all** pd deadlines uniformly by λ (EDF order unchanged):

| λ | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|-----|------|-----|-----|-----|-----|-----|
| schedulable | 457 | 619 | 708 | 784 | **810** | 793 | 776 |

So **larger ("slower") local deadlines make more systems schedulable, up to a
point** — non-monotone, with an interior optimum. (For reference: `equal`,
`period_prop` and "every task = flow.deadline" give 314, 320 and 407.)

## Research questions (work through them in order)

1. **Artifact or real?** The analysis rewards generous deadlines partly because
   its interference bound credits a task with interfering over fewer windows the
   later its local deadline. Re-optimise under the honest constraint
   `Σ d_i ≤ D_flow` (and `d_i ≤ D_flow`). How much of the ~810 survives?
2. **Order vs magnitude.** Decompose the gain: `λ` isolates the *magnitude*
   effect (order fixed); permuting deadlines isolates the *order* effect. Which
   dominates, and where do the extra systems come from (per utilization level)?
3. **Adaptive rule.** Instead of a global γ/λ, choose it per flow (by number of
   tasks, utilization, slack). Is γ*≈0.5 universal or does it track `u`/structure?
4. **Closed form vs search.** Compare closed-form rules against search
   (hill-climb / simulated annealing / gradient over deadlines): how much
   headroom is left, and at what cost (schedulables vs time)?
5. **Generalisation.** The pool is a single seed/size. Add sizes (see
   `pool.SIZES`), mappings and deadline factors; does the interior optimum
   survive, or is it an artifact of this pool?

## Verification (a separate, adversarial pass)

Your findings will be checked by an **independent verifier** in a *separate*
session (see `.opencode/commands/verify.md`). The verifier reads `NOTES.md` and
your code, tries to **falsify** each claim, and writes `VERIFICATION.md`; it does
**not** edit `NOTES.md`. Make that possible:

- Keep `NOTES.md` and the code (`algorithms/` modules, and any probe script, put
  it in `tmp/`) sufficient to **reproduce every number** you report: the exact
  command, the pool/seed, and the interpreter (the repo venv, Python 3.13).
- Make every claim **falsifiable**: state the prediction from the mechanism
  before the experiment, and the observed result after. Separate what you
  *proved* from what you only *conjectured*.
- Do not overstate. An honest "uncertain" is worth more than a claim the verifier
  refutes.

## Deliverable

Keep **`NOTES.md`** at the sandbox root up to date: for each question, the
hypothesis, the mechanism-based prediction, what you ran, the numbers and the
conclusion. Be quantitative and cite the run configuration (systems, levels,
seed). Leave the code that backs your findings in `algorithms/` (or `tmp/` for
one-off probes). `VERIFICATION.md` is written by the separate verifier, not you.
