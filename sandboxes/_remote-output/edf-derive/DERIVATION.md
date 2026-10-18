# Derivation

> Deliverable for `sandbox-edf-derive`: a **single rule** `d = g(system)` that
> assigns each task's local deadline to maximise the number of schedulable
> instances of the frozen pool (seed 42, 50 systems × 20 utilisation levels =
> **1000 instances**), judged by the holistic local-EDF analysis in
> `frozen/analysis.py`.
>
> **Reproducibility.** Everything below is produced with the repository venv,
> Python 3.13 (`../../../.venv/bin/python`), on this sandbox's frozen core, seed
> 42, size 15 (5 flows × 3 tasks, 3 processors), population 50, 20 levels in
> [0.5, 0.9] → 1000 instances. The analysis is numerically sensitive at the
> boundary, so a different interpreter shifts counts by a few instances.
> *What is derived* is marked **[D]**; *what is only located empirically* is
> marked **[O]**.
>
> **Registered rules.** `wad` (`algorithms/wad.py`, one pass, **875**) and
> `wad_adapt` (`algorithms/wad_adapt.py`, seed + 3 bounded correction passes,
> **930**). Reproduce with `run.py --per-level -q`; probes in `tmp/wad_*.py`.

---

## 0. Setup and notation

A **system** is a set of **flows**; a flow is an ordered chain of tasks mapped
onto processors; every processor runs **local EDF**. For a task `i`:

| symbol | meaning | source |
|---|---|---|
| `C_i` | worst-case execution time | pool |
| `T_i` | period (= its flow's period) | pool |
| `D_f` | the flow's **end-to-end** deadline (the hard requirement) | pool |
| `d_i` | the task's **local** deadline (**the free parameter we assign**) | rule |
| `j_i` | release jitter = `max` WCRT of the flow-predecessors (`0` first) | `Task.jitter` |
| `R_i` | worst-case response time of task `i` | analysis |

A flow is schedulable iff `flow.wcrt ≤ flow.deadline`, where `flow.wcrt` is the
response of the flow's **last** task; the system iff every flow is. So `d_i` is
**not a requirement** — it is a scheduling parameter that, under local EDF, also
fixes the priority order. Throughout, `a` is the **analysed** task and `t` ranges
over the **interferers** (other tasks on `a`'s processor).

---

## 1. Understanding the analysis (Step 0)

`frozen/analysis.py` computes `task.wcrt` as the fixed point of a holistic
response-time analysis. Below, equation by equation, and **every place a local
deadline enters**.

### 1.1 Busy period (eq 5, `_busy_period`) [D]

`L` is the least fixed point of

```
L = ceil(L/T_a)·C_a + Σ_{t≠a, same proc} ceil((L + j_t)/T_t)·C_t ,   L ≥ C_a.
```

**No local deadline appears.** `L` depends only on `(C, T, j)`; deadlines affect
it only indirectly through the jitters `j_t` (which are WCRTs). The `p`-loop
below runs `p = 1 … ceil(L/T_a)`.

### 1.2 Interference term (eq 1, `_wi`) — the crux [D]

```
pl = ceil((t + j_t)/T_t)                                   # releases in a window of length t
pd = 0                                     if D < d_t       # D = analysed window point (psi)
   = floor((j_t + D − d_t)/T_t) + 1        otherwise        # t-deadline points at or before D
m  = min(pl, pd)                                           # activations credited to t
w_t = m · C_t
```

This is the **EDF demand cap**: only interfering activations whose absolute
deadline `(q−1)T_t − j_t + d_t` is `≤ D` are counted. Two consequences we use:

* **Direction.** `d_t` enters only through `D − d_t`. A **later** interferer
  deadline ⇒ smaller `pd` ⇒ fewer credited activations ⇒ **less** interference.
* **Cutoff.** If `d_t > D` then `pd = 0` and `t` contributes **nothing** at that
  window point. Conversely, while `pd ≥ pl` the cap does not bite and increasing
  `d_t` changes nothing (the interference **saturates**).

**Probe (`tmp/wad_probe.py`, panel B).** Victim `a(C=2,T=10,d_a=2)` and
interferer `t(C=3,T=10)`: `R_a = 5` for `d_t ∈ {0,1,2}`, then drops to `2` for
`d_t ≥ 5`. Weakly decreasing with the exact cutoff, exactly as eq (1) predicts.

### 1.3 ψ sets (eq 4, 6) and the `p`-loop [D]

* eq (4) — interferers' deadline points `{(q−1)T_t − j_t + d_t} ∪ {d_t}`:
  these **slide with `d_t`**.
* eq (6) — the analysed task's deadline points `{(k−1)T_a + d_a}`: depend on
  `d_a`. (Comprehension variable shadows the argument, so this set does not
  depend on the activation index — harmless here.)
* For activation index `p`, the enumerated window is
  `[(p−1)T_a + d_a , p·T_a + d_a)` — it **slides with `d_a`**.

### 1.4 Fixed point (eq 8) and response (eq 9) [D]

```
w    = p·C_a + Σ_{t≠a} _wi(t, w, psi)        fixed point, from w = p·C_a
r    = w − psi + d_a + j_a
R_a  = max over (p, psi in p's window) of r
```

Where `d_a` appears, and the direction:

1. **Additively `+ d_a`** in eq (9) — raises `R_a`.
2. **In `psi_ab = (p−1)T_a + d_a`** and in the window bounds. At the candidate
   `psi = psi_ab` the shift and the additive term **cancel exactly**:

   ```
   r = w − ((p−1)T_a + d_a) + d_a + j_a = w − (p−1)T_a + j_a      (d_a cancels).
   ```

   So along `psi_ab` the value of `d_a` is irrelevant; it acts only by
   **re-selecting which interferer points fall in the window** and by the value
   of `pd_t(psi)` there.
3. For an interferer candidate `psi`, `r = w(psi) − psi + d_a + j_a`: raising
   `d_a` raises this term **directly** — the *cost* of a large local deadline.

### 1.5 The lever, summarised [D]

| where | effect of increasing that deadline |
|---|---|
| interferer `d_t` (eq 1) | fewer credited activations ⇒ **less** interference on `a` |
| analysed `d_a` (eq 9, `psi_ij`) | `+d_a` ⇒ **more** response for `a` |
| analysed `d_a` (eq 9, `psi_ab`) | **cancels**; only re-selects window points |

Every task is both analysed and an interferer, so a rule trades column 2 against
column 1. This is the origin of the **interior optimum** in the magnitude, seen
in §2.3.

### 1.6 Limit-factor abort [D]

While scanning `psi`, as soon as a provisional `r > D_f · limit_factor` the
analysis raises; with `reset=True` it wipes **all** WCRTs and the system counts
as unschedulable. `harness.is_schedulable` uses `limit_factor = 1`, so the test
is exactly "max provisional `r ≤ D_f` for every task". A rule that pushes one
task past `D_f` loses the **whole** system. Our correction passes read responses
with a huge `limit_factor`; the **judge** always uses `1`.

---

## 2. Restricted cases (derive the shape)

### 2.1 One processor, two tasks, no jitter [D]

Two independent single-task flows on one processor, `a(C_a=2)` and `b(C_b=3)`,
`T=10`, `j_a=j_b=0`. `tmp/wad_probe.py` panel A (entries `(R_a, R_b)`):

| `d_a \ d_b` | 2 | 4 | 6 | 8 | 10 | 20 |
|---|---|---|---|---|---|---|
| **2** | (5,5) | (3,5) | (2,5) | (2,5) | (2,5) | (2,5) |
| **4** | (5,3) | (5,5) | (3,5) | (2,5) | (2,5) | (2,5) |
| **6** | (5,3) | (5,3) | (5,5) | (3,5) | (2,5) | (2,5) |
| **8** | (5,3) | (5,3) | (5,3) | (5,5) | (3,5) | (2,5) |
| **10** | (5,3) | (5,3) | (5,3) | (5,3) | (5,5) | (2,5) |
| **20** | (5,3) | (5,3) | (5,3) | (5,3) | (5,3) | (5,5) |

Reading it off the analysis:

* If `d_a < d_b`: at `psi = d_a`, `pd_b = 0`, so `w = C_a` and the self point
  gives `R_a = C_a = 2`. The only other candidate is `psi = d_b`, where
  `w = C_a + C_b = 5` and `r = 5 − d_b + d_a`. Hence

  ```
  R_a = max( C_a , C_a + C_b − (d_b − d_a) ),        R_b = C_a + C_b = 5.
  ```

  `R_a = C_a` for **free** once the gap `d_b − d_a ≥ C_b`; otherwise a residual
  `C_a + C_b − (d_b − d_a)` remains (panel B: `5 − (d_t − 2)`).
* If `d_a = d_b`: each counts the other, both get `C_a + C_b = 5` (the diagonal).

**Two derived facts.**

1. **Ordering = priority.** The tasks that delay `a` at its self point are
   exactly those with `d_t ≤ d_a`. Smaller local deadline ⇒ higher local-EDF
   priority ⇒ not delayed by later-deadline tasks. Equal deadlines make everyone
   delay everyone.
2. **Magnitude = cap relief.** The interferer's *absolute* deadline controls the
   residual: pushing `d_b` up by 1 reduces `R_a` by 1 until saturation.

### 2.2 Add a chain / jitter [D]

Flow `t0 → t1` across two processors, plus one interferer `u` sharing `t1`'s
processor (`T=10`, `C_0=C_1=2`, `C_u=3`, `d_0=2, d_1=8`). `tmp/wad_probe.py`
panel C:

| `d_u` | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| `R_0` | 2 | 2 | 2 | 2 | 2 |
| **end-to-end `R_1`** | 7 | 7 | 5 | 4 | 4 |

Raising the interferer's deadline lowers the **end-to-end** response (7 → 4):
the eq(1) mechanism survives holistic coupling. `j_1 = R_0` is unchanged (the
interferer is not a predecessor), so this is interference, not jitter.

**Cost propagation ⇒ back-loading [D].** For a flow `0,1,…,m−1`, raising `d_k`
raises `R_k` (the self point `w(d_k)` counts more interferers), and
`R_k = j_{k+1}` is the **release jitter of the successor**. A raise at position
`k` therefore **compounds along the chain** up to the end-to-end response.
Raising the **last** task's `d_{m−1}` does **not** compound — it has no
successor. Hence the marginal *cost* of a large local deadline is **largest for
the first task and smallest for the last**, while the *benefit* (less
interference on co-processor tasks) is the same wherever it is placed. A
cost-minimising allocation therefore **back-loads**: later tasks in a flow get
larger local deadlines.

### 2.3 Several flows per processor ⇒ shape + magnitude [D direction, O constants]

With many tasks per processor the interference sets overlap and no closed form
drops out. The two levers of §2.1–§2.2 give a two-parameter family. Let the WCET
shares be `s_i = C_i·D_f/ΣC` (so `Σ_i s_i = D_f`) and `P_{i−1} = Σ_{k<i} s_k`:

```
shape:      r_i  = s_i + g·P_{i−1}                 (g ≥ 0 back-loads later tasks)
magnitude:  d_i  = r_i · λ · D_f / Σ_j r_j         (λ > 0;  Σ_i d_i = λ·D_f)
```

* `g = 0, λ = 1` is exactly `pd`; `g = 1, λ = 1` is `pd_global`.
* The directed prediction is `g > 0` (back-load) and `λ > 1` (inflate to slide
  interferer points out of the analysed windows).
* **Interior optimum in `λ`.** Reproduced on the shipped families
  (`tmp/eval_families.py`): uniform `λ` on `pd` gives 457/618/**708**/787/**810**
  /793/775 for λ = 0.5/0.75/1.0/1.5/**2.0**/3.0/5.0; the γ family
  (pd→pd_global interpolation) peaks at ~814 around γ ≈ 0.3–0.5. `equal` 314,
  `allD` 407. This is exactly the §1.5 tension: below λ ≈ 2 the caps do not
  bite; above it the analysed task's own `+d_a` push and the cap stop paying.
* The constants are **located empirically**. Over the 2-D grid
  `a·cum_i + b·C_i` (`tmp/batch.py`, `tmp/batch2.py`) the plateau is broad; the
  best linear point is `≈ 1.4·C_i + 0.3·cum_i` (866–868).

**Refinement of the shape [O].** Comparing a *prior-work* ramp
`cum_{i−1}/ΣC = P_{i−1}/D_f` against a *position* ramp `i/(n−1)`
(`tmp/batch5.py`, `tmp/batch7.py`, four seeds), the **position ramp is slightly
more robust**: `d_i = D_f (Q C_i/ΣC + S·i/(n−1))` peaks at `Q ≈ 1.4–1.6`,
`S ≈ 0.3–0.4` with 868–875 across seeds, versus 855–868 for the prior-work form.
The two encode the same idea (later ⇒ larger) but the position ramp does not
penalise a flow whose *last* task happens to be small. We adopt it.

---

## 3. The rule

One fixed function of the system; the same code runs for every instance.

### 3.1 Seed — the one-pass closed form (`algorithms/wad.py`)

For a flow with `n` tasks, total `ΣC = Σ_j C_j`, task index `i = 0…n−1`:

```
d_i = D_f · ( Q · C_i / ΣC  +  S · i / (n−1) ),      Q = 1.6,  S = 0.4
```

(For `n = 1`, the position term is set to `1.0`.) Two ingredients:

* **Proportional term** `Q·C_i/ΣC`: a task that executes more must be allowed
  more time, scaled by a safety factor `Q > 1` because its response exceeds its
  execution (it also carries interference).
* **Position ramp** `S·i/(n−1)`: earlier tasks get smaller local deadlines
  (higher local-EDF priority), keeping their responses — and the jitter of their
  successors — small; the ramp hands the remaining end-to-end budget to later
  tasks (§2.2). `S = 0` collapses to a uniform scale of `pd` and loses the chain
  structure; `Q = 0` ignores the execution spread. Both terms are needed.

One pass, no analysis, no search. **875/1000** on the shipped pool (seed 42).

### 3.2 Correction passes (`algorithms/wad_adapt.py`) — the main rule

The seed uses **one** magnitude for the whole system, but flows differ (wcet
spread, deadline factor, which processor they map onto), so for some flows the
seed is too generous and for others too tight — the residual the search headroom
exposes (§4). Correct it per flow. Repeat `P = 3` times:

```
analyse the system with limit_factor = ∞        # read untruncated responses R_f
for each flow f:
    phi_f = clip( (D_f / R_f)^α , [lo, hi] )
    scale every local deadline of f by phi_f

α = 0.5,  [lo, hi] = [0.75, 1.25].
```

Rationale **[D]**:

* `D_f < R_f` (flow late) ⇒ `phi_f < 1`: its local deadlines are too large for
  its own response ⇒ **shrink them**; this lowers its own response (the `+d_a`
  and cap terms of §1.5).
* `D_f > R_f` (flow has slack) ⇒ `phi_f > 1`: it can afford **larger**
  deadlines, which (eq 1) suppress the interference it imposes on its
  processor-mates without violating its own deadline.
* `α < 1` **damps** the step (the response is not linear in the deadlines), and
  the **clip is a trust region**: a large downward step on one flow raises
  interference on its processor-mates, so the update must stay local. The
  corrected deadlines are **not** required to sum to `D_f`.

This is a **fixed count** (3) of passes — not trial-and-error over candidates.

### 3.3 Assumptions and complexity

**Assumptions.** (A1) The pool's structure — chains mapped round-robin, local
EDF on coupled processors — with the frozen holistic analysis as the judge.
(A2) A flow may "spend" more than its budget, `Σ_i d_i > D_f`, since the local
deadline is not a requirement; the seed spends `≈ 2.2·D_f` per flow. (A3) The
correction passes may **read** the system's own responses; the brief permits
reading the system and allows "a fixed small number of passes; no search".

**Cost.** `wad`: one pass, 0 analyses, ~1e-5 s/instance. `wad_adapt`:
`P = 3` analyses of the full system, ~0.023 s/instance — dominated by the judge's
own analysis.

**Non-iterative?** In the intended sense, yes: a closed form plus a fixed count
(3) of correction passes, with no per-instance hand-tuning.

---

## 4. Validation

Frozen pool, seed 42, 1000 instances, repo venv Python 3.13. Exact command:

```
../../../.venv/bin/python run.py --per-level -q
```

| technique | schedulable | note |
|---|---|---|
| `pd` (shipped) | **708** | baseline |
| `pd_global` (shipped) | **744** | cumulative, budget-feasible |
| best γ family (reproduced) | ~814 | partial globalisation |
| best λ family (reproduced) | ~810 | uniform scale of `pd` |
| **`wad`** (one pass) | **875** | §3.1 |
| **`wad_adapt`** (this rule) | **930** | §3.2 |
| own coordinate-descent headroom | ~974 | lower bound, `tmp/wad_headroom.py` |

`pd`/`pd_global` reproduce the shipped values exactly. The headroom is my own
1-pass coordinate descent (multiplicative per-task moves, tardiness objective),
run on a **25-system subset** (500 instances): 487/500 ⇒ ≈ 974/1000 equivalent;
the AGENTS-quoted reference is ~898. It is a **lower bound** on the per-instance
optimum.

**Per-utilisation breakdown** (`run.py --per-level -q`):

| u | pd | pd_global | `wad` | **`wad_adapt`** |
|---|---|---|---|---|
| 0.500–0.584 | 50 | 50 | 50 | 50 |
| 0.605 | 47 | 49 | 49 | 50 |
| 0.626 | 47 | 48 | 49 | 50 |
| 0.647 | 47 | 48 | 48 | 50 |
| 0.668 | 46 | 46 | 48 | 50 |
| 0.689 | 44 | 44 | 48 | 50 |
| 0.711 | 39 | 43 | 48 | 49 |
| 0.732 | 39 | 42 | 48 | 48 |
| 0.753 | 33 | 37 | 47 | 48 |
| 0.774 | 31 | 32 | 44 | 46 |
| 0.795 | 27 | 28 | 43 | 45 |
| 0.816 | 20 | 23 | 40 | 45 |
| 0.837 | 15 | 19 | 36 | 44 |
| 0.858 | 11 | 17 | 30 | 37 |
| 0.879 | 9 | 11 | 27 | 36 |
| 0.900 | 6 | 7 | 20 | 32 |
| **Σ** | **708** | **744** | **875** | **930** |

Gains appear where interference binds (u ≳ 0.65) and grow with load: at u = 0.9
the rule makes 32/50 schedulable versus 6 for `pd`.

**Seed robustness** (`wad_adapt`, other pool seeds — the metric pool is seed 42):

| seed | 1 | 7 | 42 | 123 |
|---|---|---|---|---|
| `wad` /1000 | 862 | 855 | 875 | 868 |
| `wad_adapt` /1000 | 927 | 919 | 930 | 928 |

So the rule is not a seed-42 fit: it stays 919–930 across seeds.

> **Note on co-located work.** Other techniques are registered in this sandbox
> from a parallel session (`bl_scale` 865 one-pass, `bl_adaptive` 912, a
> back-loaded seed with two correction passes). They agree on the mechanism
> (back-load + inflate + bounded per-flow correction) and are reported by
> `run.py` alongside these; the rules above were derived and calibrated
> independently here.

---

## 5. Validity domain and failure cases

**Near-optimal at low load.** For u ≤ 0.6 everything is schedulable for any
reasonable rule, so nothing is gained there. The correction closes almost all of
the mid-range gap: at u ∈ [0.6, 0.8] the rule is within 0–2 of the headroom.

**Where it fails (u ≥ 0.84).** The remaining gap to the ~974 headroom is
concentrated at high load. There the binding structure is **per-task and
erratic** — the optimiser finds wildly different per-task deadlines, and a
flow-level correction cannot capture an intra-flow imbalance (e.g. one task
badly placed while the *flow-level* response looks fine). Evidence:

* A **per-task** correction (every task scaled by its own response) was *worse*
  than the per-flow one at the same settings — the per-task signal is noisy and
  the collateral interference on processor-mates dominates.
* A **too-wide** trust region is destructive; `[0.4, 4.0]` collapses the score,
  confirming the response is strongly non-linear in the deadlines.
* More passes do not help beyond P = 3 (P = 2 ≈ 923, P = 3 ≈ 926–930,
  P = 4 ≈ 928); the rule converges quickly.

**Assumption (A2) is load-bearing.** The seed overspends each flow budget
(`Σ_i d_i ≈ 2.2·D_f`). A budget-feasible variant of the family is weaker
(`pd_global` 744; the budget-normalised shape at λ = 1 reaches only ~762). Part
of the gain therefore comes from spending **beyond** `D_f`. This is a property of
the **bound** (§1.2), not of the real system, and the pool imposes no budget
constraint, so it is legitimate under the stated objective — but it must be
flagged.

---

## 6. Discussion

**Is this exploiting the approximate analysis?** Yes, deliberately and openly.
The analysis credits an interferer with fewer activations the later its local
deadline (eq 1). The rule lifts local deadlines to exploit exactly that. In a
real system, preemption does not depend in this way on another task's *local*
deadline (the local deadline only orders the *local* EDF queue). The reference
bound is the stated objective, so optimising it is the task; but the effect is a
property of the bound, not a claim about real hardware.

**Proved vs observed.**

* *Derived:* §1.2 (direction/cutoff), §1.4 (`d_a` cancels at `psi_ab`),
  §2.1 (two-task table and the residual formula), §2.2 (cost propagation ⇒
  back-load), §2.3 (interior optimum in λ as the cost/benefit tension), §3.2
  (the direction and the need for damping/trust region).
* *Observed / located:* the constants `Q = 1.6, S = 0.4, α = 0.5,
  [0.75, 1.25], P = 3`; the exact counts; the ~974 headroom (a lower bound).
  The constants sit on broad plateaus and the rule is seed-robust, which is
  evidence they are not a seed-42 fit — but they are not derived in closed form.

**Generalisation.** The direction (back-load + inflate, then correct per flow)
follows from the analysis and should carry to other sizes/mappings/deadline
factors; only the constants should shift. This sandbox exposes only `size = 15`,
so a cross-size study is out of scope; the seed-robustness table is the available
evidence. **Open questions:** (i) a closed form for `λ` derived from the busy
period instead of a swept constant; (ii) a derived trust region rather than
`[0.75, 1.25]`; (iii) whether the residual at u ≥ 0.84 is inherent to any
flow-level rule.

**Reproduce.**

| command | output |
|---|---|
| `../../../.venv/bin/python run.py --per-level -q` | §4 tables |
| `../../../.venv/bin/python tmp/wad_probe.py` | §1.2 cap, §2.1 table, §2.2 chain |
| `../../../.venv/bin/python tmp/batch.py`, `tmp/batch2.py` | §2.3 (shape, magnitude) grid |
| `../../../.venv/bin/python tmp/batch5.py`, `tmp/batch7.py` | position vs prior-work ramp |
| `../../../.venv/bin/python tmp/wad_adapt.py`, `tmp/wad_adapt2.py` | correction-pass sweep, seed check |
| `../../../.venv/bin/python tmp/wad_headroom.py` | headroom (population 25) |

Registered rules: `algorithms/wad.py` (`wad`, 875), `algorithms/wad_adapt.py`
(`wad_adapt`, 930). Scratch scripts in `tmp/`.
