# EDF Gradient Descent: Surrogate, Vectorized, and Exact Approaches

This document summarises the investigation into EDF local deadline optimisation
using gradient-based methods.  All code lives under `code/`.

---

## 1. Background

The goal is to assign per-task deadlines for **Local EDF** scheduling such that
the system becomes schedulable.  Three gradient-descent strategies were
explored:

| Strategy | Gradient source | Speed | Accuracy |
|---|---|---|---|
| **Sequential FD** | Finite differences on scalar `HolisticLocalEDFAnalysis` | 1× (gold standard) | 100 % |
| **Surrogate** | Autograd on a soft differentiable model | 13× per grad | ~87 % schedulability |
| **Vectorized V2** | Finite differences on batched exact analysis | 7–8× per grad | ~94 % of Seq gap |

---

## 2. Sequential FD (reference)

```python
from gradient_descent.gradient_function import SequentialGradientFunction
```

- Calls `HolisticLocalEDFAnalysis.apply()` **2n times** (central differences).
- Each call runs the full scalar EDF analysis.
- **Pro**: exact gradient.  **Con**: slow — O(n²) per optimizer iteration.

---

## 3. Differentiable Surrogate

**Files:** `surrogate/surrogate_edf.py`  
**Development:** `workspace/surrogate_edf_local/`

### Approach

A fully-differentiable PyTorch model of the Holistic Local EDF equations:

- Soft ceil/floor via sigmoid at half-integer boundaries.
- Fixed uniform ψ grid (M = 50 points per job p).
- Unrolled w_ab fixed point (N_w = 10 iterations).
- Softmax over (p, ψ) for the worst-case response time.
- Autograd to obtain ∇cost w.r.t. sigmoid-encoded deadlines.

### Results

| Metric | Value |
|---|---|
| WCRT correlation (Pearson r) | 0.82 |
| Gradient cosine vs Seq | 0.31 |
| % positive gradient | 90 % |
| Schedulability (8 systems) | 7/8 (87 %) |
| Speed-up per gradient | 13× |
| Speed-up full optimisation | ~2× |

### Why it falls short

- The uniform ψ grid misses the critical absolute-deadline points where
  interference changes discontinuously.
- The soft ceil/floor approximations blur the gradient.
- The surrogate cost (logsumexp) differs from the real cost (hard max).

**Conclusion:** Viable as a proof-of-concept but not competitive with exact
methods for schedulability.

---

## 4. Vectorized Exact Analysis – V1 (grid-based)

**File:** `vector/vector_edf.py`

### Approach

Batched exact EDF analysis using a fixed ψ grid (M = 100):

- Tensor shape `(scenarios, tasks, P_max, M)`.
- Exact ceil/floor arithmetic (NumPy, not differentiable).
- Same cache + early-termination architecture as `vector_fp.py`.

### Bug discovered

- **`D_limit` used task deadlines instead of flow deadlines** for the
  over-limit check, causing premature termination and completely wrong WCRTs.
- Fixed by adding `D_flow` (flow-level deadlines) separately.

### Results (after fix)

| Metric | Value |
|---|---|
| WCRT error vs scalar | ~4 % (grid discretisation) |
| Gradient cosine vs Seq | 0.45 |
| % positive gradient | 100 % |
| Schedulability (8 systems) | 8/8 (100 %) |
| Speed-up full optimisation | 2× |

---

## 5. Vectorized Exact Analysis – V2 (exact ψ set)

**File:** `vector/vector_edf_v2.py`  
**Status:** ✅ Canonical version

### Approach

Instead of a uniform ψ grid, evaluates at the **exact same absolute-deadline
points** used by the scalar analysis:

- **Own deadlines:** (p−1)·T_i + D_i
- **Interfering deadlines:** q·T_j − J_j + D_j, filtered by interval and
  non-negative release time.
- **Pure deadlines:** D_j (scalar adds `{t.deadline for t in tasks}`).

Architecture:
- Tasks processed sequentially in topological order (intra-iteration jitter
  propagation).
- Vectorised over **scenarios** (s = 2n+1 for finite differences).
- `_compute_w` helper for the w_ab fixed point.

### Bugs found and fixed

1. **J_init used predecessor WCET, not cumulative WCRT.**
   The scalar `init_wcrt` sets `task.wcrt = task.wcet + predecessor.wcrt`.
   `task.jitter` (a property) returns `predecessor.wcrt`.  V2 initially used
   `J_init[i] = C[pred]` instead of the cumulative value.
   → Fixed by computing cumulative WCET in `_get_vectors`.

2. **Cost function used task deadlines instead of flow deadlines.**
   `InvslackCost` computes `max((flow.wcrt − flow.deadline) / flow.deadline)`.
   V2 used `max((R[i] − D[i]) / D[i])` over all 12 tasks.
   → Fixed by using `last_idx` (last task of each flow) and `flow.deadline`.

3. **Busy period L computed once with J_init, never updated.**
   The scalar recomputes `_busy_period` each iteration with the current
   jitter.  As WCRTs increase, jitter increases, L increases, and more
   interfering ψ values are considered.
   → Fixed by recomputing L inside the outer loop with `J_ref = max(J, axis=0)`.

### Remaining limitation

6 out of 24 perturbed scenarios yield WCRTs that **do not match** the scalar
analysis.  V2 underestimates WCRTs for these scenarios (by 13–112 units).
The root cause appears to be in jitter propagation between scenarios within
the outer convergence loop — the perturbed scenario's jitter is not fully
updated in certain conditions.

**Practical impact:** The optimisation achieves **110/140 (78.6 %)** vs the
sequential gold standard **113/140 (80.7 %)** — a gap of only 3 systems out
of 140.

### Results

| Metric | Value |
|---|---|
| WCRT match vs scalar (base) | max diff 2.6×10⁻⁵ |
| WCRT match vs scalar (perturbed) | 18/24 scenarios exact |
| Gradient cosine vs Seq | 0.56 |
| Schedulability (8 systems) | 8/8 (100 %) |
| Schedulability (140 eval) | 110/140 (78.6 %) |
| Speed-up full optimisation | 1.7–1.8× |
| Speed-up per gradient | 7–8× |

---

## 6. Vectorized Exact Analysis – V3 (per-scenario L)

**File:** `vector/vector_edf_v3.py`  
**Status:** ⚠️ Experimental — same results as V2, slower

### Approach

Computes the busy period L **per scenario** (`_busy_period_multi` with
broadcasting over s), rather than using max-jitter.  This was intended to
eliminate over-estimation from conservative L.

### Result

V3 gives the **same WCRTs as V2** (same bug inherited) and is **slower**
(0.28 s vs 0.16 s) due to the (s, n, n) tensor operations in
`_busy_period_multi`.  Not recommended over V2.

---

## 7. Comparative Evaluation (SchedRatioEval)

20 systems, 12 utilisation levels (0.5–0.9), EDF scheduling, PD initialisation.

### Schedulability (u ≤ 0.72, 7 levels × 20 = 140)

| Method | Sched | Ratio | vs PD | Time |
|---|---|---|---|---|
| **PD** (Deadline Monotonic) | 59 | 42.1 % | — | 0.00 s |
| GDPA-Surr (surrogate) | 69 | 49.3 % | +7 pp | 1.38 s |
| **GDPA-Vec V2** | **110** | **78.6 %** | **+36 pp** | **1.42 s** |
| HOPA | 92 | 65.7 % | +23 pp | 1.33 s |
| GDPA-Seq (gold) | 113 | 80.7 % | +39 pp | 1.85 s |

### V2 vs HOPA head-to-head

| U | PD | HOPA | **Vec V2** | Winner |
|---|---|------|---------|--------|
| 0.50 | 16 | 20 | 20 | tie |
| 0.54 | 13 | 18 | **19** | Vec |
| 0.57 | 11 | 17 | **19** | Vec |
| 0.61 | 7 | 15 | **18** | Vec |
| 0.65 | 6 | 10 | **15** | Vec |
| 0.68 | 4 | 6 | **12** | Vec |
| 0.72 | 2 | 6 | **7** | Vec |

**V2 beats HOPA at 6/7 utilisation levels.**  The gradient-based method
consistently outperforms the specialised EDF heuristic.

### Gap analysis

- PD → Seq gap: 54 systems.
- V2 captures **51/54 = 94.4 %** of the gap.
- HOPA captures 33/54 = 61.1 %.
- Surrogate captures 10/54 = 18.5 %.

---

## 8. Key Technical Insights

### 8.1 Why the surrogate underperforms

The EDF scheduling decision depends on comparing absolute deadlines (ψ vs Dⱼ).
This comparison is a step function — discontinuous.  The surrogate smooths it
with a sigmoid, which blurs the interference boundary.  The uniform ψ grid
misses the critical points where the step occurs.

FP surrogates face a similar issue with priority ordering, which is also
discrete.  However, EDF's deadline parameter is **continuous**, making a
surrogate *conceptually* more viable than for FP.  The practical challenge is
that the set Ψ is dynamic and the worst-case occurs at specific discontinuity
points.

### 8.2 Why vectorized EDF is harder than vectorized FP

| FP | EDF |
|---|---|
| Priority matrix `(s, t, t)` — fixed boolean | Interference depends on ψ (continuous parameter) |
| All tasks share same number of w-iterations | Different (task, p) pairs have different ψ sets |
| No ψ dimension | Need 4D tensors `(s, t, P, M)` or per-task ψ construction |

### 8.3 The critical role of the busy period

The busy period L_i determines how many jobs (p values) and interfering
deadlines (q values) to consider.  L_i depends on jitter J_j of all tasks
on the same processor.  Jitter propagates across flows (predecessor WCRT)
and across outer iterations.

**Failing to recompute L each iteration** was the single biggest bug —
it caused V2 to miss 50 % of schedulable systems (80 → 110 after fix).

### 8.4 Jitter initialisation

`task.jitter` is a **property** returning `max(predecessor.wcrt)`.
After `init_wcrt`, WCRTs are cumulative (task i's WCRT includes all
predecessors' WCETs).  So jitter = predecessor's **cumulative WCET**, not
just its WCET.  Using `C[pred]` instead of the cumulative value caused
significant WCRT errors.

### 8.5 Cost function alignment

`InvslackCost` computes cost over **flows** (end-to-end deadlines), not
individual tasks.  The vectorized analysis must use `flow.deadline` and
only the **last task** of each flow for the cost computation.

---

## 9. File Map

```
code/
├── surrogate/
│   └── surrogate_edf.py          ← differentiable surrogate + gradient
├── vector/
│   ├── vector_fp.py              ← vectorized FP analysis (reference)
│   ├── vector_edf.py             ← V1: grid-based vectorized EDF
│   ├── vector_edf_v2.py          ← V2: exact ψ set (CANONICAL)
│   └── vector_edf_v3.py          ← V3: per-scenario L (experimental)
├── analysis/
│   └── holistic_local_edf_analysis.py  ← scalar EDF analysis
├── gradient_descent/
│   ├── gradient_function.py      ← SequentialGradientFunction
│   └── gradient_optimizer.py     ← GradientDescentOptimizer
└── workspace/
    ├── surrogate_edf_local/      ← surrogate development & validation
    └── edf_vector_evaluation/    ← SchedRatioEval results & scripts
```

---

## 10. Usage

### Vectorized gradient descent (V2)

```python
from vector.vector_edf_v2 import VectorEDFGradientFunctionV2
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import DeadlineExtractor
from gradient_descent.cost_functions import InvslackCost
from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis

analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
ph = DeadlineExtractor()
cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
grad_fn = VectorEDFGradientFunctionV2(sigma=1.5, cost_limit_factor=10)

optimizer = GradientDescentOptimizer(
    parameter_handler=ph,
    cost_function=cost_fn,
    stop_function=...,
    gradient_function=grad_fn,
    update_function=...,
)
optimizer.apply(system)
```

### Surrogate gradient descent

```python
from surrogate.surrogate_edf import SurrogateEDFGradient

grad_fn = SurrogateEDFGradient(tau=0.05, N_w=10, N_jitter=2, M_psi=50)
```

### Standalone vectorized EDF analysis

```python
from vector.vector_edf_v2 import VectorHolisticEDFAnalysisV2, ResultsCache

vec = VectorHolisticEDFAnalysisV2(limit_factor=10, cache=ResultsCache())
vec.apply(system)  # sets task.wcrt for the base system

# With additional scenarios:
vec.apply(system, scenarios_deadlines=D_array)  # D_array shape (s, n)
R = vec.scenarios_response_times  # shape (n, s)
```
