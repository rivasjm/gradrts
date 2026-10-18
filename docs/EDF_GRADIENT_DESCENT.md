# EDF deadline optimisation: analyses and approaches

This document describes the **local-EDF deadline-assignment** stack in `code/`:
the response-time analysis, the gradient/descent methods built on it, and a
vectorized analysis for evaluating many deadline assignments at once.

---

## 1. The problem

Assign each task's **local deadline** (`task.deadline`) so that the system is
schedulable under **local EDF**. Every processor runs EDF (local clocks); a
task's interference depends on the absolute-deadline points (the set the code
calls `psi`) of its processor-mates, and release jitter propagates across the
flow (a task's jitter is its predecessor's WCRT). The task -> processor mapping
is fixed; the local deadlines are the only knob.

---

## 2. Scalar analysis (the reference)

**File:** `analysis/holistic_local_edf_analysis.py` -> `HolisticLocalEDFAnalysis`

For one system with one set of deadlines it computes each task's worst-case
response time (WCRT). It is *holistic*: jitter coupling is solved by an outer
fixed point, tasks are processed in `system.tasks` order, and the response time
of a task is the max over the candidate `psi` points in each busy-period window.

It is **the reference** used to score any assignment: the judge runs it with
`limit_factor=1`, while search methods may use a higher `limit_factor` so that
near-infeasible states still return usable values.

---

## 3. Sequential finite-difference gradient

**File:** `gradient_descent/gradient_function.py` -> `SequentialGradientFunction`

Central differences over the scalar analysis (**2n** analysis calls per
gradient). This is what the paper's EDF scenario
(`workspace/framework_paper/edf/edf.py`) uses, together with `DeadlineHandler`
(`gradient_descent/parameter_handlers.py`), `InvslackCost` and
`GradientDescentOptimizer`. Exact gradient, but O(n^2) per optimizer iteration.

---

## 4. Vectorized exact analysis

**File:** `vector/vector_edf.py` -> `VectorHolisticEDFAnalysis`

Evaluates a whole `(K, n_tasks)` batch of local-deadline assignments **at once**
with numpy, and reproduces the scalar analysis **bit for bit** (comparison-based
window assignment, CPython-compatible compensated summation, expression-order
matching). One batched call is several times faster than `K` scalar calls.

```python
from vector.vector_edf import VectorHolisticEDFAnalysis

wcrts = VectorHolisticEDFAnalysis().apply(system, deadlines)   # (K, n_tasks)
# deadlines[j][i] is the local deadline of task i (system.tasks order) for
# assignment j; the result is (K, n_tasks) WCRTs (inf where the analysis aborts).
```

The class takes an optional **`itcap`**: `None` (default) is the exact unbounded
analysis; an integer caps the outer Gauss-Seidel loop and drops scenarios that
abort or fail to converge within `itcap` sweeps (bit-identical to the unbounded
analysis on converging scenarios, and robust as a *search* objective — the
unbounded loop can need thousands of sweeps for badly infeasible assignments).

---

## 5. Assignment algorithms

| Class | File | Uses |
|---|---|---|
| `PDAssignment` | `assignment/assignments.py` | proportional deadlines (baseline) |
| `HOPAssignment` | `assignment/hopa_assignment.py` | iterative heuristic (scalar) |
| `RTACDAssignment` | `assignment/rta_cd_assignment.py` | scalar response-time-guided coordinate descent |
| `VecILSAssignment` | `assignment/vec_ils_assignment.py` | batched multi-start local search (uses `VectorHolisticEDFAnalysis`) |

`RTACDAssignment` starts from an analytic assignment (tight-flow priority) and
runs a coordinate descent on the local deadlines using the analysis itself as
the objective, short-circuiting as soon as the `limit_factor=1` analysis reports
the system schedulable.

`VecILSAssignment` scores a **whole neighbourhood** of candidate deadline vectors
per call with the vectorized analysis: best-improvement coordinate descent over
a geometric grid of multiplicative moves, wrapped in an iterated local search
(perturbed restarts) under a wall-clock budget.

(The fixed-priority `SwapILSAssignment` in `assignment/swap_ils_assignment.py`
belongs to the FP stack; see `FP_GRADIENT_DESCENT.md`.)

---

## 6. Historical investigation (removed)

Earlier vectorized EDF variants and a differentiable surrogate were explored and
have since been removed from the tree; the key findings are kept below.

- **Vectorized V1 / V2 / V3** (files `vector/vector_edf.py` grid-based,
  `vector/vector_edf_v2.py` exact `psi` set, `vector/vector_edf_v3.py`
  per-scenario busy period) — **removed**, superseded by the bit-exact
  `vector/vector_edf.py`.
  - V2 matched the scalar analysis within ~1e-4 (not bit-exact) and
    underestimated WCRTs on some perturbed scenarios.
  - V1 used a fixed `psi` grid (~4 % error); V3 produced the same values as V2
    but was slower.
- **Differentiable surrogate** (`surrogate/surrogate_edf.py`, soft ceil/floor +
  autograd over an unrolled fixed point) — **removed**. It reached ~87 %
  schedulability and a ~13x per-gradient speed-up, but never beat the exact
  methods (see §7.1).

---

## 7. Key technical insights

### 7.1 Why surrogates underperform

The EDF scheduling decision compares absolute deadlines (`psi` vs `D_j`) — a
step function. A sigmoid smoothing blurs the interference boundary, and a
uniform `psi` grid misses the critical points where the step occurs. (FP
surrogates hit the same wall through the priority-ordering indicator.)

### 7.2 Vectorized EDF is harder than vectorized FP

| FP | EDF |
|---|---|
| Fixed boolean priority matrix `(s, t, t)` | Interference depends on `psi` (continuous) |
| All tasks share the same number of `w` iterations | Different `(task, p)` pairs have different `psi` sets |
| No `psi` dimension | A `(s, t, P, M)` tensor, or per-task `psi` construction |

The bit-exact implementation in `vector/vector_edf.py` avoids the `psi`-grid
tensor: the candidate `psi` *values* do not depend on the outer index `p`, so the
nested `(p, psi)` loop collapses into a single max per candidate, and the scalar
arithmetic is reproduced expression-for-expression.

### 7.3 The critical role of the busy period

The busy period `L_i` determines how many jobs (`p` values) and interfering
deadlines (`q` values) to consider; it depends on the jitter of all tasks on the
processor, which propagates across flows (predecessor WCRT) and across outer
iterations. Failing to recompute `L` each iteration was the single biggest bug in
the old V2 — it missed ~50 % of schedulable systems.

### 7.4 Jitter initialisation

`task.jitter` is a **property** returning the predecessor's WCRT. After
`init_wcrt`, WCRTs are cumulative, so jitter = the predecessor's **cumulative
WCET**, not just its WCET.

### 7.5 Cost function alignment

`InvslackCost` measures over **flows** (end-to-end deadlines), not individual
tasks: use `flow.deadline` and only the **last task** of each flow.

---

## 8. File map

```
code/
├── analysis/
│   └── holistic_local_edf_analysis.py   scalar EDF analysis (the reference)
├── vector/
│   ├── vector_fp.py                     vectorized FP analysis
│   └── vector_edf.py                    vectorized EDF analysis (bit-exact)
├── gradient_descent/
│   ├── gradient_function.py             SequentialGradientFunction
│   ├── parameter_handlers.py            DeadlineHandler
│   ├── cost_functions.py                InvslackCost
│   └── gradient_optimizer.py            GradientDescentOptimizer
├── assignment/
│   ├── assignments.py                   PDAssignment (baseline)
│   ├── hopa_assignment.py               HOPAssignment (heuristic)
│   ├── rta_cd_assignment.py             RTACDAssignment (scalar descent)
│   └── vec_ils_assignment.py            VecILSAssignment (batched search)
└── workspace/framework_paper/edf/       EDF scenario (paper evaluation)
```

---

## 9. Usage

### Sequential FD gradient (the paper's EDF scenario)

```python
from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import SequentialGradientFunction
from gradient_descent.parameter_handlers import DeadlineHandler

analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
handler = DeadlineHandler()
cost_fn = InvslackCost(parameter_handler=handler, analysis=analysis)
grad_fn = SequentialGradientFunction(cost_function=cost_fn)

PDAssignment().apply(system)          # starting point
# optimizer = GradientDescentOptimizer(parameter_handler=handler,
#                                      cost_function=cost_fn,
#                                      gradient_function=grad_fn, ...)
# optimizer.apply(system)
```

### Batched vectorized analysis

```python
from vector.vector_edf import VectorHolisticEDFAnalysis

vec = VectorHolisticEDFAnalysis()                  # exact
vec = VectorHolisticEDFAnalysis(itcap=1000)        # bounded (search objective)
R = vec.apply(system, deadlines)                   # (K, n_tasks)
```

### Assignment algorithms

```python
from assignment.rta_cd_assignment import RTACDAssignment
from assignment.vec_ils_assignment import VecILSAssignment

RTACDAssignment(budget=20.0).apply(system)
VecILSAssignment(improve_budget=30.0, max_total=48.0).apply(system)
```
