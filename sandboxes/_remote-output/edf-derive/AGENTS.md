# Real-time systems: deriving a non-iterative deadline-assignment rule

You are an agent working inside a small, self-contained sandbox. It is a study in
**derivation**: given the holistic local-EDF response-time analysis, **derive a
rule that assigns local deadlines** so that **the greatest possible number of
example systems is schedulable** — ideally an algorithm that is
**non-iterative** (or a few passes). The single metric is the count of
schedulable instances (see *Objective* below).

The evaluation core (model, analysis, pool, harness) is **shared, verbatim, with
the sibling sandbox `sandbox-edf-techniques`**, so the numbers are directly
comparable: same pool (seed 42), same interpreter, same baselines. The only new
tool here is an **instrumented** analysis that shows *why* a response time is
what it is (see below).

## Isolation (work only inside this sandbox)

This sandbox is a **standalone** study. Work **only inside this directory**: do
not read, write, list or import anything from other sandboxes — in particular
the sibling `sandbox-edf-techniques`, which is a **separate, live** study — nor
anywhere else on the filesystem. Depending on external files breaks the
comparability of the numbers and their reproducibility (that sibling changes
while you work and is not part of your task). If you need a search/optimiser as
a headroom reference, **implement it yourself** under `algorithms/` or `tmp/`.

## The problem

A *system* is a set of **flows** (ordered chains of tasks) mapped onto
**processors**; every processor runs **local EDF**. Each task has a **local
deadline** `task.deadline` — the free parameter a rule assigns — and each flow
has a fixed **end-to-end deadline** `flow.deadline`. A system is *schedulable*
iff every flow meets its end-to-end deadline under the holistic analysis in
`frozen/analysis.py`.

Concretely, the analysis is a **forward map** `d ──► R`: you give it local
deadlines and it returns worst-case response times, via a max over activation
windows of fixed points, with a holistic jitter fixed point on top. The task here
is the **inverse**: produce `d = g(system)` that makes `R_i ≤ flow.deadline` for
all flows, (near-)optimally.

## Objective (the metric)

The goal is concrete and single: applying your rule to the pool
(`frozen/pool.py`: **50 example systems** swept over **20 utilization levels** =
**1000 instances**), **make as many of those instances schedulable as possible**.
That is exactly the `schedulable` column of `harness.evaluate()` / `run.py` —
maximise it. Report it as the total (out of 1000) and per utilization level.

- **Targets to beat:** `pd` = **708**, `pd_global` = **744**; the best simple
  families reach ~**815** (partial globalisation, γ≈0.5) and ~**810** (uniform
  scale, λ≈2). A per-system optimiser (search / gradient, cf. the paper's `gdpa`)
  reaches higher (~898) but is **iterative and per-instance** — it is the
  **headroom reference**, not the shape of the answer we want.
- **One rule for all systems.** `g` is a **single fixed function** applied to
  every instance: no per-system hand-tuning, no hard-coded answers. It may read
  the system (wcets, periods, flow deadlines, mapping) but the *same code* must
  run for every system.
- A rule that is **explainable** and reaches, say, 810 is worth more than a fit
  that scores 815 on this one pool and breaks elsewhere.

## Quick start

From the sandbox root (this directory):

```bash
python run.py                 # every registered technique: 50 systems x 20 levels = 1000 instances
python run.py --per-level     # per-utilization breakdown
python run.py -q              # quiet
```

```python
from frozen import harness, pool, explain, format_detail

report = harness.evaluate()                   # every technique registered in algorithms/
report["techniques"]["pd"]["schedulable"]     # 708

system = pool.get_systems(population=1)[0]    # one example system
pool.set_utilization(system, 0.8)
ok, details = explain(system, limit_factor=1000)   # instrumented breakdown
print(format_detail(system.tasks[-1].name, details[system.tasks[-1].name]))
```

### Interpreter (reproducibility)

Run everything with the repository virtualenv — `../../../.venv/bin/python`
(Python 3.13). The analysis is **numerically sensitive** near the schedulability
boundary: another interpreter (e.g. the system `python3`, 3.9) shifts the counts
by a few instances.

## Layout

```
frozen/analysis.py      the judge: holistic local-EDF analysis
frozen/instrumented.py  the SAME analysis + per-task interference breakdown
                        (binding p*/psi*, per-interferer m, whether the deadline
                        cap binds) -- your main derivation tool
frozen/model.py         Task / Flow / Processor / LinearSystem
frozen/pool.py          deterministic example pool (seed 42): 5 flows x 3 tasks x 3 procs
frozen/harness.py       evaluate(...): counts schedulable instances per technique
algorithms/             YOUR workspace: one module per rule
algorithms/pd.py        pd / pd_global (the shipped baselines)
run.py                  the runner
```

## Registering a technique

Add a module under `algorithms/` and decorate the callable with `@register`
(the function name is the label); every module is imported automatically:

```python
from algorithms import register

@register
def my_rule(system):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        for t in flow.tasks:
            t.deadline = ...      # what the analysis judges
```

Rules: **do not modify `frozen/` or `run.py`**; keep new work in `algorithms/`;
put scratch scripts in `tmp/` (create it).

## Step 0 — understand the holistic analysis *in detail* (do this first)

You cannot invert what you have not understood. Study `frozen/analysis.py` and
`frozen/model.py` line by line, and **write your understanding in
`DERIVATION.md`** before deriving anything. Cover at least:

- **The model.** A task's release **jitter** is the max WCRT of its
  flow-predecessors → processors are *coupled*; the analysis iterates to a fixed
  point. Schedulability is the **flow's end-to-end** deadline; the local deadline
  is **not** a requirement, only a scheduling parameter (it also fixes the EDF
  order).
- **The interference term `w_i(t, D)` (eq 1)** — the crux. Find exactly where the
  interfering task's local deadline `d_i` enters, and why `D < d_i` contributes
  nothing. This is the **demand cap**.
- **The `psi` sets (eq 4, 6)**, the busy period (eq 5), the `p`-loop.
- **The fixed point `w_ab` (eq 8)** and `r_ab = w_ab − psi + d_a + j_a` (eq 9);
  enumerate every place a local deadline appears (analysed task vs interferer)
  and its direction. Note that at the worst `psi = (p−1)T_a + d_a`, `d_a`
  cancels in the response.
- **The `limit_factor` abort** (`harness.is_schedulable` uses `limit_factor=1`).

Use `frozen/instrumented.py` to *check* your understanding on real systems: for a
task, see which interferer's cap binds and how much each contributes.

The analysis is explicitly **approximate**. Say where in your derivation you rely
on that, and where the rule could be "exploiting the model" rather than the real
system.

## The observation to explain (the target to beat)

The local deadline is a free parameter, and **larger ("slower") local deadlines
help, up to a point**:

| γ (partial globalisation) | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| schedulable | 708 | 773 | 804 | 813 | 813 | **815** | 803 | 786 | 773 | 758 | 744 |

| λ (uniform scale of pd deadlines) | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 | 5.0 |
|---|---|---|---|---|---|---|---|
| schedulable | 457 | 619 | 708 | 784 | **810** | 793 | 776 |

Non-monotone, **interior optimum**; `pd` = 708, `pd_global` (γ=1) = 744, while
`equal` / `period_prop` / "every task = flow.deadline" give only 314 / 320 / 407.
So the best simple formula is *not* an extreme.

## The derivation task

Deliver a rule `d = g(system)` and its derivation. Suggested path (adapt freely):

1. **Restricted cases first.** Start tiny: one processor, two tasks, no jitter.
   Invert the (now explicit) condition to get the *shape* of the optimal
   assignment — e.g. equalise a demand budget, or set the caps so the interference
   in the critical window is balanced. Record the exact derivation.
2. **Add the pieces back** one at a time: jitter on one link; a chain (holistic
   coupling); several flows sharing a processor. See how the rule must change.
3. **State the rule and its assumptions** explicitly, and write it as a
   technique in `algorithms/`. Prefer a **non-iterative** formulation (one pass,
   or a fixed small number of passes; no search).
4. **Validate** on the pool (`harness.evaluate` / `run.py`): compare against
   `pd`, `pd_global`, the γ/λ families and, as a headroom reference, whatever a
   search/optimiser reaches (you may implement one to know the ceiling). Report
   schedulables and where the rule fails.
5. **Characterise the validity domain** using `frozen/instrumented.py`:
   in which systems/levels is it near-optimal, and when does the cap structure
   break the assumption?

Keep it honest: the goal is a *rule you can justify*, not a number. A rule that
is 810 and explainable is worth more than a fit that is 815 on this one pool.

## Verification (a separate, adversarial pass)

Your derivation will be checked by an **independent verifier** in a *separate*
session (see `.opencode/commands/verify.md`), writing `VERIFICATION.md` and **not**
editing `DERIVATION.md`. So: keep everything reproducible (exact commands,
pool/seed, interpreter), keep every claim falsifiable (assumption → prediction →
observation), and separate what you *derived* from what you only *observed*.

## Deliverable

- **`DERIVATION.md`** — assumptions, the derivation (the maths, in steps), the
  final rule, and the validation numbers (with the failing cases).
- The rule as a module in `algorithms/` (registered), plus any probe scripts in
  `tmp/`. `VERIFICATION.md` is written by the separate verifier, not you.
