# FP Gradient Descent: Exact, Vectorized, and Surrogate Approaches

This document synthesises the investigation into Holistic Fixed-Priority (FP)
priority optimisation using gradient-based methods.  All code lives under
`code/`.

---

## 1. Background

The goal is to assign per-task **fixed priorities** for distributed real-time
systems such that the system becomes schedulable under the Holistic FP
analysis of Tindell et al.  The optimisable parameters are the task priorities
— inherently **discrete** (ordinal) — which makes gradient-based methods
non-trivial.

Several gradient-descent strategies were explored, ranging from exact
finite-difference gradients over the true analysis to fully differentiable
PyTorch surrogates.

| Strategy | Gradient source | Sched (/2000) | vs DM | Speed |
|---|---|---|---|---|
| **GDPA** | Finite differences on vectorized exact Holistic | **1741** | **+511** | 123 s |
| HOPA (heuristic) | Hill-climbing ka/kr | 1687 | +457 | 31 s |
| V1-FD | Finite differences on linearized V1 surrogate | 1555 | +325 | 495 s |
| V1-TopK | V1 ranking + Holistic validation | 1398 | +168 | 620 s |
| V3-opt / V1-unroll / V1-anneal / V1-implicit | Autograd on soft-priority surrogate | 1230 | +0 | 94–2275 s |
| DM (baseline) | Deadline Monotonic | 1230 | — | 0.4 s |

**GDPA** (Gradient Descent Priority Assignment) is the only non-dominated
method.  Every approach that attempts to make the FP analysis differentiable
through soft-priority relaxation fails to improve over Deadline Monotonic.

---

## 2. The Holistic FP Analysis

### 2.1 Algorithm (`HolisticFPAnalysis`, `analysis/holistic_fp_analysis.py`)

Four nested loops:

```
while WCRTs not converged (outer):     # depends on jitter from predecessor WCRT
    for each task:
        hp = higher_priority(task)      # same-processor, higher-priority tasks
        limit = deadline × limit_factor
        for p = 1, 2, ...:              # busy period jobs
            while w not converged:       # fixed-point iteration
                w = Σ⌈(J_j + w) / T_j⌉ × C_j  +  p × C_i
                r = w − (p−1) × T_i + J_i
                if r > task.wcrt: task.wcrt = r
                if r > limit: abort      # unschedulable
            if w ≤ p × T_i: break        # busy period closed
```

The analysis is *holistic* because jitter (task release jitter) propagates
across processors: a task's jitter equals its predecessor's WCRT.  The outer
convergence loop solves this circular dependency globally.

### 2.2 Sources of non-differentiability

All five touch the parameter being optimised (priorities):

| Source | Location | Nature |
|---|---|---|
| `higher_priority(task)` | `analysis_function.py:23` | **Indicator function** of priority ordering. Changes discretely when priorities cross. |
| `math.ceil((J + w) / T)` | `holistic_fp_analysis.py:37` | Step function. Gradient 0 almost everywhere, undefined at integers. |
| `max(r, task.wcrt)` | `holistic_fp_analysis.py:42` | Not differentiable at equality (subdifferentiable otherwise). |
| `while w != w_prev` | `holistic_fp_analysis.py:35` | Variable loop count changes computation graph between calls. |
| `if r > limit: return` | `holistic_fp_analysis.py:44` | Early return changes which tasks are analysed. |

The **most critical** is `higher_priority` because it is the direct
dependency between priorities (the parameter) and the analysis.  Without
relaxing this indicator function, the analysis is **locally constant** in
the parameter — small changes to priority values that do not change the
ordering produce zero gradient.

---

## 3. GDPA — Vectorized Finite Differences over Exact Holistic

**File:** `vector/vector_fp.py`

### 3.1 Architecture

- `VectorHolisticFPAnalysis` (l.167): a batched version of the Holistic FP
  analysis using NumPy 3D tensors.  Accepts a stack of priority matrices
  `(scenarios, tasks, tasks)` and computes WCRTs for all scenarios in a
  single pass.

- `VectorFPGradientFunction` (l.59): computes the gradient via central
  finite differences using the vectorized analysis.  For `x` of length `n`:
  - Perturb each parameter by ±δ (2n scenarios).
  - Batch-evaluate all 2n scenarios through `VectorHolisticFPAnalysis`.
  - `∇cost_i ≈ (cost(x+δeᵢ) − cost(x−δeᵢ)) / 2δ`.

- `ResultsCache` (l.9): caches intermediate results by priority matrix
  key.  As the optimisation progresses, many priority matrices re-appear
  across iterations (same perturbation patterns).  The cache prunes
  known scenarios before each evaluation.

### 3.2 Why it works

- The gradient uses **finite differences**, not autograd.  The analysis is
  treated as a black box — discontinuous points are "crossed" by the
  perturbation, yielding a valid finite-difference slope.

- The **vectorization** means all 2n scenarios share a single execution of
  the outer/inner convergence loops.  Cost per gradient step ≈ 10 µs per
  scenario in cached regime.

- The `PriorityExtractor` (in `parameter_handlers.py`) encodes priorities
  through `sigmoid(priority)`, mapping them to a continuous space where
  perturbations have physical meaning.  The priority matrix is built from
  comparisons (`priorities < priorities.T`), so the analysis remains
  discrete — but GDPA does not require differentiability.

### 3.3 Results

| Metric | Value |
|---|---|
| Schedulable systems (/2000) | 1741 |
| vs DM | +511 |
| vs HOPA | +54 |
| Total time (100 systems, 20 U levels) | 123 s |
| Frontier | **Dominant** |

GDPA outperforms HOPA despite being an unguided gradient descent against a
specialised heuristic.  It is the gold standard for FP priority optimisation.

---

## 4. Linearized Surrogates (V1, V2, V3)

**File:** `workspace/holistic_linearization/linearized_fp.py`

### 4.1 Motivation

The Holistic FP analysis is expensive per evaluation in scalar (Python) mode.
The idea: replace the inner `while w not converged` fixed-point loop with a
closed-form expression, creating a fast surrogate correlated with the true
analysis.

### 4.2 The linearization

Replaces `⌈(J + w) / T⌉` by `(J + w) / T + α`:

```
w = (p×Cᵢ + α×ΣCⱼ + ΣJⱼ×uⱼ) / (1 − Σuⱼ)      where uⱼ = Cⱼ/Tⱼ
```

No more fixed-point iteration.  The p-loop and outer convergence remain.

### 4.3 Versions

| Version | Outer loop | p-loop | Nature | Spearman vs Holistic |
|---|---|---|---|---|
| V1 | Yes (convergence) | Yes | Iterative | 0.94 |
| V2 | No (N passes) | Yes | Semi-iterative | 0.71 |
| V3 | No | No (p=1, matrix solve) | Algebraic pure | 0.75 |

**V1** is the most accurate surrogate (Spearman 0.94 with Holistic).  V3
is the only one that is trivially differentiable (matrix solve) but is the
least accurate.

### 4.4 Key insight

V1 with limited iterations (5–10) still maintains Spearman > 0.83 while
being 20–40× faster than scalar Holistic.  However, once vectorized
(§10), V1 costs ≈ Holistic per evaluation (~10 µs with cache), negating
the speed advantage.

---

## 5. Differentiable Surrogates (Soft-Priority Relaxation)

**Files:** `workspace/holistic_linearization/differentiable_v3.py`,
`differentiable_v1.py`

### 5.1 V3 soft-priority (`differentiable_v3.py`)

Replaces the hard `higher_priority` set with a sigmoid-softened mask:

```python
hp_soft[i, j] = σ((sⱼ − sᵢ) / τ)  ×  same_proc_mask
```

Where `sᵢ` are learnable priority scores and `τ` is a temperature
parameter (lower = sharper, closer to hard priority).

Uses V3's algebraic formulation (`(I − M) r = b`, p=1).  The priority
scores enter `hp_soft`, which enters `M` and `D`, making the **entire
pipeline differentiable** w.r.t. scores via PyTorch autograd.

**Components integrated into `GradientDescentOptimizer`:**
- `GradientFunction`: `V3SoftPriorityGradient` (autograd)
- `CostFunction`: `InvslackCost` (Holistic FP real, for stop condition)
- `ParameterHandler`: `RawPriorityHandler` (continuous priorities ≥ 0)
- `UpdateFunction`: `NoisyAdam` (Adam + decaying Gaussian noise)

**Result: +0 systems over DM.**  The gradient guides the scores in the
continuous landscape, but the resulting priority *orderings* do not
transfer to the Holistic analysis.

### 5.2 V1 unrolled / annealed / implicit (`differentiable_v1.py`)

Three phases attempting to make V1 differentiable:

| Phase | Technique | Mechanism |
|---|---|---|
| 1 | Unrolled V1 | N V1 iterations unrolled in PyTorch with soft-priority and soft p-loop via softplus. Fully differentiable. |
| 2 | Annealing | Temperature τ decays: `τ(t) = max(τ_min, τ₀ × decayᵗ)`. Starts smooth, sharpens toward discrete. |
| 3 | Implicit differentiation | Solves `r* = F(r*, θ)` to convergence, uses the implicit function theorem for exact gradient at the fixed point. |

**All three → +0 over DM.**  The time cost is 248–2275 s with zero
improvement.

### 5.3 Why soft-priority relaxation fails

The INFORME conclusion (§7) is categorical:

> *Cualquier relajación diferenciable del orden de prioridades (softmax,
> sigmoid, annealing) produce gradientes que optimizan el surrogate pero
> no transfieren al Holistic real. El problema es la discrepancia entre
> el paisaje continuo de scores y el espacio discreto de asignaciones.*

**Three root causes:**

1. **Landscape mismatch.**  The continuous "score" landscape optimised by
   autograd is fundamentally different from the discrete "ordering"
   landscape evaluated by Holistic.  A local minimum in score-space
   (∇surrogate ≈ 0) does not correspond to a schedulable priority ordering.

2. **Indicator-function gradient.**  Even with sigmoid relaxation, the
   gradient propagates through a narrow sigmoid window where `sⱼ ≈ sᵢ`.
   Once priorities separate, the sigmoid saturates and the gradient
   vanishes — but the priority *order* can still be suboptimal.  The
   soft-priority gradient is most informative precisely when the surrogate
   is *least* accurate (τ high), and vanishes when it becomes *most*
   accurate (τ low).

3. **The surrogate ≠ the real analysis.**  V3 (p=1, one-shot) has
   Spearman 0.75 with Holistic.  V1 (iterative, p-loop) has 0.94, but
   making V1 differentiable required unrolling/annealing/implicit
   differentiation — and *none* of these transferred to Holistic.
   The limitation is not the formula's accuracy but the mapping between
   continuous scores and discrete orderings.

### 5.4 The EDF contrast

EDF deadline optimisation *does* benefit from differentiable surrogates
(see `EDF_GRADIENT_DESCENT.md`).  This is because EDF's parameter
(deadline) is **continuous** and directly enters the analysis.  FP's
parameter (priority) is **ordinal** — only the relative ordering matters,
and that ordering is a discontinuous function of the scores.

---

## 6. V1-FD — Finite Differences over V1 Surrogate

**File:** `workspace/holistic_linearization/v1_fd_gradient.py`

### 6.1 Approach

Uses V1 (Spearman 0.94) as the surrogate for **finite-difference** gradient
computation, with **discrete priorities** (no softmax):

```
GDPA:   ∂cost/∂x ≈ [Holistic(x+δ) − Holistic(x−δ)] / 2δ   (vectorized)
V1-FD:  ∂cost/∂x ≈ [V1(x+δ) − V1(x−δ)] / 2δ               (sequential)
```

Unlike the differentiable surrogates, V1-FD uses the same discrete
priority encoding as GDPA — the finite difference crosses order-toggling
boundaries and produces informative gradients.

### 6.2 Results

| Method | Sched | Time | vs DM | vs GDPA |
|---|---|---|---|---|
| V1-FD | 1555 | 495 s | +325 | −186 |

V1-FD is the **first surrogate-based method that improves over DM**
(+26.4% more schedulable systems).  However, it is dominated by GDPA
(+41.5%) because V1-FD is sequential (24 independent V1 evaluations per
step) rather than vectorized (1 batched pass).

---

## 7. V1-TopK — Ranking + Validation

**File:** `workspace/holistic_linearization/v1_topk.py`

### 7.1 Approach

Non-iterative, two-phase:

1. **Generate** N candidate priority assignments (uniform random or
   DM-anchored swaps).
2. **Rank** all N candidates with a single `VectorLinearizedV1Analysis`
   call (V1 vectorized).
3. **Validate** the top K candidates with `HolisticFPAnalysis` and return
   the first schedulable one.

### 7.2 Results

| Method | Sched | Time | vs DM |
|---|---|---|---|
| V1-TopK (N=100, K=10, pert=3) | 1398 | 620 s | +168 |

V1-TopK fails to beat HOPA (+457) and is dominated by V1-FD (+325).

### 7.3 Why it fails

The INFORME §12 identifies two causes:

1. **Cost parity.**  V1 vectorized costs ≈ Holistic vectorized (~10 µs/eval
   with cache).  N V1-evals + K Holistic-evals ≈ (N+K) Holistic-evals,
   comparable to or worse than GDPA's 720 Holistic-evals in absolute cost.

2. **Coverage, not identification.**  The search space is combinatorial
   (e.g. 4!³ = 13824 permutations for a 4t×3p system).  Sampling N=100
   candidates covers <1% of the space.  V1 ranks them correctly, but
   none may be schedulable for heavily-loaded systems.  HOPA and GDPA
   succeed because they *move* through the space with structure (local
   search, gradient), not because they evaluate better.

> *Un ranking perfecto sobre muestras mediocres sigue dando resultados
> mediocres. La calidad de la exploración domina a la calidad de la
> evaluación cuando el espacio de búsqueda es combinatorio.*

---

## 8. Comparative Results

100 systems 3f×4t×3p, utilisation sweep U ∈ [0.5, 0.9] (20 levels, 2000
total evaluations per method).

### 8.1 Main table

| Method | Sched | Time (s) | vs DM | Frontier |
|---|---|---|---|---|
| DM | 1230 | 0.4 | — | efficient (ref) |
| HOPA | 1687 | 31 | +457 | efficient |
| **GDPA** | **1741** | **123** | **+511** | **dominant** |
| V1-FD | 1555 | 495 | +325 | dominated |
| V1-TopK | 1398 | 620 | +168 | dominated |
| V3-opt | 1230 | 94 | +0 | dominated |
| V1-unroll | 1230 | 249 | +0 | dominated |
| V1-anneal | 1230 | 248 | +0 | dominated |
| V1-implicit | 1230 | 2275 | +0 | dominated |

### 8.2 By utilisation level

At low U (≤ 0.6), all methods perform similarly.  GDPA separates from
HOPA at U > 0.65.  All differentiable surrogates collapse to DM across
the entire range — they never find a better ordering than Deadline
Monotonic.

GDPA captures **94.1%** of the upper bound (1741/1849 of schedulable
systems at maximum).

---

## 9. Framework Architecture

### 9.1 `gradient_descent/` — optimisation framework

| Module | Role |
|---|---|
| `interfaces.py` | ABCs: `ParameterHandler`, `CostFunction`, `GradientFunction`, `UpdateFunction`, `StopFunction` |
| `gradient_optimizer.py` | `GradientDescentOptimizer`: wires all components together, loops x ← x + update(∇cost(x)) |
| `gradient_function.py` | `SequentialGradientFunction`, `AvgSeparationDelta`, finite-difference helpers |
| `cost_functions.py` | `InvslackCost(handler, analysis)`: inserts x into system, runs analysis, returns max flow slack |
| `parameter_handlers.py` | `PriorityExtractor`, `DeadlineExtractor`, `MappingPriorityExtractor`, `MappingOnlyExtractor` |
| `update_functions.py` | `Adam`, `GradientNoise`, `NoisyAdam` |
| `stop_functions.py` | `ThresholdStopFunction` (cost < ε), patience-based stops |

### 9.2 `vector/` — batched/vectorized analyses

| Module | Role |
|---|---|
| `vector_fp.py` | `VectorHolisticFPAnalysis` (exact, batched), `VectorFPGradientFunction` (GDPA), `ResultsCache`, `PrioritiesMatrix`, `MappingPrioritiesMatrix`, `MappingOnlyMatrix` |

### 9.3 `analysis/` — scalar analyses

| Module | Role |
|---|---|
| `holistic_fp_analysis.py` | `HolisticFPAnalysis` — the canonical scalar Holistic FP analysis |
| `holistic_global_edf_analysis.py` | Global EDF variant |
| `holistic_local_edf_analysis.py` | Local EDF variant |

### 9.4 `model/` — data model

| Module | Role |
|---|---|
| `linear_system.py` | `LinearSystem`, `Task`, `Flow`, `Processor` |
| `analysis_function.py` | `AnalysisFunction` ABC, `higher_priority`, `init_wcrt`, `reset_wcrt`, `LimitFactorReachedException` |

### 9.5 Optimisation loop flow

```
ParameterHandler.extract(system)  →  x  (list of floats, e.g. sigmoid-encoded priorities)

while not stop:
    cost = CostFunction.compute(system, x)          # insert x, run Holistic, measure slack
    grad = GradientFunction.compute(system, x)       # perturb x, batch-evaluate, FD gradient
    update = UpdateFunction.update(system, x, grad, t)  # Adam + noise
    x ← x + update
    ParameterHandler.insert(system, x)  →  x ← ParameterHandler.extract(system)  (normalise)

solution = StopFunction.solution(system)  # best x found
```

---

## 10. Vectorization of V1 (WP-1)

**File:** `workspace/holistic_linearization/vector_v1.py`

`VectorLinearizedV1Analysis` replicates the API of
`VectorHolisticFPAnalysis` but uses V1's closed-form `w` instead of the
Holistic fixed-point iteration.

- Numerical equivalence with scalar V1: 2-norm diff < 1e-13.
- Spearman vs Holistic: 0.978 (pooled over 50 systems).
- **Cost per evaluation (with cache): ≈ 10 µs** — same order of magnitude
  as `VectorHolisticFPAnalysis`.

**Conclusion:** Vectorizing V1 does not create a speed advantage.  The
bottleneck in Holistic (scalar) was the inner `while w not converged`
loop; vectorization eliminates that bottleneck for both V1 and Holistic,
equalising their cost.  V1 remains valuable for **ranking** but not for
**faster evaluation**.

---

## 11. Key Technical Insights

### 11.1 The ordinality wall

FP priority optimisation faces a fundamental barrier not present in EDF
deadline optimisation: the parameter (priority) is **ordinal**, not
cardinal.  The analysis depends on `higher_priority(task)` — an indicator
function of the ordering.  Any relaxation that makes this function
differentiable (sigmoid, softmax) introduces a landscape mismatch between
the surrogate and the true analysis.

This does not affect EDF because EDF's parameter (deadline) is continuous
and enters the interference calculations directly (absolute deadline
comparisons).

### 11.2 Finite differences cross the wall

Finite-difference gradients (GDPA, V1-FD) work because the perturbation
is large enough to **toggle priority orderings**, producing a
non-zero cost difference.  The analysis itself is never made
differentiable — only the outer optimisation loop is.

Autograd-based gradients (V3-opt, V1-unroll/anneal/implicit) fail because
they require the *inner* analysis to be differentiable.  The soft
relaxation's gradient lives in a different landscape.

### 11.3 Vectorization equalises surrogate and exact cost

The initial motivation for surrogates (V1) was speed: Holistic scalar was
slow due to Python `while` loops.  Vectorization eliminates those loops
for both V1 and Holistic, making their evaluation costs comparable.
The surrogate is only faster in *ranking* (one eval = one rank), not in
*evaluation* (one eval ≈ same time).

### 11.4 Exploration quality > ranking quality

V1-TopK's failure teaches an important lesson: a near-perfect ranker
(Spearman 0.97) over mediocre samples still produces mediocre results.
The bottleneck is generating good candidates, not evaluating them.
GDPA succeeds because the finite-difference gradient provides an
**informed direction** that moves all priorities simultaneously toward
the optimum.

### 11.5 The priority matrix as a bridge

All FP gradient methods share a common abstraction: the **priority
matrix** `(tasks, tasks)`, a boolean matrix where `pm[i, j] = True` if
task j has higher priority than task i on the same processor.

- In GDPA, it is built from the perturbed `x` values via comparison.
- In V1-FD, the same construction.
- In V3-opt, it is replaced by a sigmoid-softened matrix.

The priority matrix is the **only** way priorities enter the analysis.
This makes it the natural abstraction boundary between "discrete ordering"
and "continuous optimisation".

---

## 12. Viable Paths Forward

### Open

| Path | Status | Description |
|---|---|---|
| **GDPA tuning** | Open | Optimise init strategy, σ schedule, stop criteria, batch size, noise decay. The dominant method may still have headroom. |
| **Hybrid cache transfer** | Open | The `ResultsCache` accumulates results across iterations. Transferring hot scenarios between steps could reduce total Holistic evaluations. |
| **V1-FD vectorized** | Open | Replace sequential V1 evaluations with batched `VectorLinearizedV1Analysis`. May make V1-FD time-competitive with GDPA while preserving the gradient quality. |
| **GDPA + HOPA hybrid** | Open | Use GDPA for global direction + HOPA local search on the final assignment. |

### Closed

| Path | Reason |
|---|---|
| **Soft-priority differentiable surrogate** | Landscape mismatch (§5.3, §11.1). All four attempts (V3-opt, V1-unroll, V1-anneal, V1-implicit) produced +0 over DM. |
| **V1-TopK random sampling** | Coverage, not identification (§7.3). Sampling-based exploration does not scale combinatorially. |
| **Surrogate-assisted GD (V3 direction + Holistic acceptance)** | The soft-priority gradient does not point toward Holistic-improving directions (§5.1). A line-search on Holistic cost would reject all steps. |

---

## 13. File Map

```
code/
├── analysis/
│   └── holistic_fp_analysis.py      ← canonical scalar Holistic FP analysis
├── vector/
│   └── vector_fp.py                 ← VectorHolisticFPAnalysis + GDPA (CANONICAL)
├── gradient_descent/
│   ├── interfaces.py                ← ABCs: ParameterHandler, CostFunction, ...
│   ├── gradient_function.py          ← SequentialGradientFunction, FD helpers
│   ├── gradient_optimizer.py         ← GradientDescentOptimizer
│   ├── cost_functions.py             ← InvslackCost
│   ├── parameter_handlers.py         ← PriorityExtractor, DeadlineExtractor, ...
│   ├── update_functions.py           ← Adam, NoisyAdam
│   └── stop_functions.py             ← ThresholdStopFunction
├── model/
│   ├── linear_system.py             ← LinearSystem, Task, Flow, Processor
│   └── analysis_function.py         ← AnalysisFunction ABC, higher_priority, etc.
├── examples/
│   └── evaluation.py                ← SchedRatioEval (comparative benchmarking)
├── tests/
│   ├── test_gradient_fp.py          ← Integration test: GDPA
│   └── test_holistic_fp_analysis.py ← Unit test: scalar Holistic FP
└── workspace/
    └── holistic_linearization/      ← DEVELOPMENT — all surrogate experiments
        ├── INFORME.md               ★ comprehensive Spanish report
        ├── linearized_fp.py         ← V1, V2, V3 (linearized analyses)
        ├── differentiable_v3.py     ← V3 soft-priority in PyTorch
        ├── differentiable_v1.py     ← V1 unrolled/annealed/implicit
        ├── v3_gradient.py           ← V3SoftPriorityGradient
        ├── v1_gradients.py          ← V1 differentiable gradients
        ├── v1_fd_gradient.py        ← V1FiniteDifferenceGradient ★
        ├── v1_topk.py               ← V1-TopK ranking + validation
        ├── vector_v1.py             ← VectorLinearizedV1Analysis
        ├── benchmark_vector_v1.py   ← V1vec vs Holvec speed comparison
        └── *.png, *.xlsx, *.csv     ← Experiment outputs
```

---

## 14. Usage

### GDPA (recommended)

```python
from vector.vector_fp import VectorFPGradientFunction, PrioritiesMatrix
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import PriorityExtractor
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from analysis.holistic_fp_analysis import HolisticFPAnalysis

analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
ph = PriorityExtractor()
cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
grad_fn = VectorFPGradientFunction(
    scenarios_builder=PrioritiesMatrix(),
    sigma=1.5,
    cost_limit_factor=10,
)
stop_fn = ThresholdStopFunction(threshold=0.0, patience=20)
update_fn = NoisyAdam(lr=3, gamma=0.9, seed=1)

optimizer = GradientDescentOptimizer(
    parameter_handler=ph,
    cost_function=cost_fn,
    gradient_function=grad_fn,
    stop_function=stop_fn,
    update_function=update_fn,
    verbose=False,
)
optimizer.apply(system)
```

### V1-FD (experimental, surrogate-based)

```python
from workspace.holistic_linearization.v1_fd_gradient import V1FiniteDifferenceGradient
from workspace.holistic_linearization.linearized_fp import LinearizedFPAnalysis

v1 = LinearizedFPAnalysis(limit_factor=10, max_p=100)
grad_fn = V1FiniteDifferenceGradient(surrogate=v1, sigma=1.5)
```

### Standalone Holistic FP

```python
from analysis.holistic_fp_analysis import HolisticFPAnalysis

analysis = HolisticFPAnalysis(limit_factor=10, reset=False, verbose=False)
analysis.apply(system)
# system.tasks[*].wcrt contains worst-case response times
```
