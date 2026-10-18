# Notes

> Local-EDF deadline-assignment study. Every number below was produced with the
> repo venv (Python 3.13) on the frozen pool `frozen/pool.py` (seed 42,
> population 50, size 15, 20 utilization levels in [0.5, 0.9] → 1000 instances),
> unless a different seed/size is stated. `task.deadline` is the free parameter;
> `flow.deadline` (end-to-end) is the only hard requirement.

---

## Analysis (Step 0 — reading `frozen/analysis.py`, `frozen/model.py`)

### 1. The model (`model.py`)

- A `LinearSystem` is a set of `Flow`s; each flow is an ordered chain of `Task`s
  mapped onto `Processor`s; every processor runs **local EDF**.
- A task `i` has: `wcet` `C_i`, `period` `T_i` (= its flow's period), a **local
  deadline** `d_i = task.deadline` (the parameter a technique assigns), and a
  release **jitter** `j_i` = `max` WCRT of its flow-predecessors (`0` for the
  first task; `model.Task.jitter`).
- `flow.wcrt` = WCRT of the flow's **last** task; the flow is schedulable iff
  `flow.wcrt is not None and flow.wcrt <= flow.deadline`. So
  **schedulability is a property of the end-to-end deadline**; the local
  deadline `d_i` is *not* a constraint, only a scheduling parameter (under EDF it
  determines the local priority order `d_i`).
- Processors are **coupled**: `jitter` of a task = WCRT of its predecessor, so
  `analysis.apply` iterates `_task_analysis` over all tasks until no `wcrt`
  changes (a holistic fixed point).
- `utilization` of a task = `C_i/T_i`; a processor's utilization is the sum. The
  pool rescales WCETs to hit a target utilization per processor.

### 2. The analysis (`analysis.py`) — equations and where deadlines enter

Notation: analysed task `a`; interfering tasks `i` are the *other* tasks on
`a`'s processor.

**Busy period, eq (5)** — `_busy_period(task, l)`:
fixed point `l = ceil(l/T_a)·C_a + Σ_{i≠a on proc} ceil((l+j_i)/T_i)·C_i`,
seeded at `l=C_a`. It counts, for the whole processor, the work that can be
released in a window of length `l`. **No local deadline appears here** (only
`C`, `T`, `j`). So the busy period is fixed once (C,T,j) are fixed; deadlines
influence it only indirectly, because `j_i` depends on predecessor WCRTs, which
*do* depend on deadlines.

**Interference term, eq (1)** — `_wi(i, t, D)`:

```
pl = ceil((t + j_i)/T_i)                                   # max # releases in window t
pd = 0                          if D <  d_i
   = floor((j_i + D - d_i)/T_i) + 1     otherwise          # # activation *deadline points* <= D
m  = min(pl, pd)
wi = m * C_i   (0 if m<=0)
```

`pd` counts the activation indices `q≥1` with `(q-1)T_i - j_i + d_i ≤ D`, i.e.
the interferer's absolute **deadline points** at or before `D`. This is the EDF
bound: only work whose deadline precedes the analysed release point can delay
it. **This is the single most important place a local deadline enters:** `d_i`
appears as `D - d_i`, so

  * a **larger** `d_i` makes `pd` **smaller** ⇒ fewer credited activations
    `m = min(pl,pd)` ⇒ **less** interference — *weakly monotone decreasing*;
  * extreme case `d_i > D` ⇒ `pd = 0` ⇒ the interferer contributes **zero**
    interference for that window point;
  * saturation: while `pd ≥ pl` (deadline points are still plentiful within the
    window) increasing `d_i` changes nothing; the benefit only starts once
    `pd < pl`, i.e. once the interferer's deadlines are pushed far enough that
    they are no longer all counted.

So "larger (slower) local deadlines help" is, in this analysis, literally
"an interferer with a later local deadline is credited with fewer interfering
activations." It is a **property of the interference bound**, not of the real
system behaviour.

**ψ sets.**
- eq (4) `_build_set_psi_ij`: release/deadline points of the interferers,
  `∪_i {(q-1)T_i - j_i + d_i : (q-1)T_i - j_i ≥ 0} ∪ {d_i}` — these sample the
  interferers' deadline points and **depend on `d_i`** (they slide with `d_i`).
- eq (6) `_build_set_psi_ab`: `{(k-1)T_a + d_a}`. *Note the comprehension
  variable `p` shadows the argument `p`, so this set is independent of `p`* — it
  is the full set of `a`'s deadline points. It **depends on `d_a`**.
- The window actually enumerated for activation index `p` is
  `[ (p-1)T_a + d_a , p T_a + d_a )` — it slides with `d_a`.

**Fixed point, eq (8)** — `_wab(task, ψ, p, w)`:
`w = p·C_a + Σ_{i≠a} _wi(i, w, ψ)`, iterated from `p·C_a`. `w` is the
busy-window length seen by the `p`-th activation and depends on `ψ` through `pd`
in `_wi`.

**Response, eq (9)** — `_ra(task, ψ, w) = w - ψ + d_a + j_a`.

Here is the subtle part. `d_a` appears **additively** (`+d_a`) *and* through the
enumerated `ψ` (the window is shifted by `d_a`; every `ψ_ab` equals
`(k-1)T_a + d_a`). For the candidate `ψ = ψ_ab = (p-1)T_a + d_a`:

```
r = w - ((p-1)T_a + d_a) + d_a + j_a = w - (p-1)T_a + j_a    → d_a cancels exactly.
```

So for `a`'s *own* deadline points the additive `+d_a` is cancelled by the
shifted window. `d_a` still matters because (i) the window slides over the
fixed interferer points `ψ_ij`, deciding *which* of them are considered, and
(ii) those `ψ_ij` feed `pd_i(ψ)`. For a `ψ_ij` candidate,
`r = w(ψ) - ψ + d_a + j_a`; as `ψ` and `d_a` shift together by `Δ` the
`-ψ + …` largely cancels, while `w` grows with `pd_i(ψ)` (which grows with `ψ`).
Because `C_i < T_i` (utilization < 1), the `-ψ` growth generally beats the `w`
growth, so on balance the analysed task's own deadline shift is not the main
driver; the driver is the interferer channel.

**Limit-factor abort.** While maximising `max_r`, as soon as any provisional
`r > flow.deadline · limit_factor` the analysis raises
`LimitFactorReachedException`; with `reset=True` all WCRTs are wiped and the
system is unschedulable. `harness.is_schedulable` uses `limit_factor=1`, so the
test is *exactly* "max provisional `r ≤ D_flow` for every task" (an early abort
and a `> D_flow` finish are equivalent). A technique that pushes deadlines so
late that some provisional `r` exceeds `D_flow` fails the whole system.

### 3. Why larger local deadlines can help — mechanism summary

1. For every task that is *someone else's interferer*, a larger `d_i` shrinks
   `pd_i` (eq 1) ⇒ fewer counted activations ⇒ smaller response times of the
   tasks it interferes with.
2. Making an interferer's `d_i` exceed the analysed release point `D` removes it
   entirely from that window.
3. The analysed task's **own** `d_a` enters eq (9) but cancels against the
   shifted `ψ_ab` window; it mainly acts by selecting which interferer deadline
   points fall in the window. Its direct `+d_a` push is largely neutralised.
4. Both effects mean that *uniformly inflating* deadlines (the `λ` family) or
   *cumulating* them (`pd_global`, γ-family) reduces credited interference —
   **even though the inflated deadlines no longer sum to the flow budget**
   (`Σ_i d_i > D_flow`), which is precisely why the honest-budget question (Q1)
   matters.
5. Very large deadlines eventually hurt (non-monotonicity): the abort in (2.)/
   the limit-factor test bites, and/or the sliding window mis-aligns with the
   interferers' discrete deadline points (a sawtooth in `d_a`).

### 3b. Micro-probe confirmation (`tmp/probe_wi.py`)

Run on system 0 at `u=0.75`, after assigning `pd` and analysing (so jitters are
populated):

- **eq(1) direction confirmed.** For interfering `task1_0`
  (`C=30.12, T=188.4, j=0`), `_wi(t=50, D)` with `D=200` is `30.12` for
  `d_i∈{0,1,10,100}`, then collapses to `0` once `d_i=500>D`. For `D=1000` it
  stays `30.12` until `d_i>1000`. So a later interferer deadline weakly
  **reduces** the counted interference to zero — the mechanism.
- **Interferer channel confirmed.** Sweeping one task's `d` on proc0 changes the
  WCRTs of *other* proc0 tasks (e.g. `task1_0`: 168.2 → 92.1 as the swept
  task's deadline grows), and non-monotonically (there is an intermediate bump).
- **`d_a` cancellation confirmed.** For the `ψ_ab` candidate `ψ=(p-1)T_a+d_a`,
  `_ra = w-(p-1)T_a+j_a` exactly for every `d_a` (the printed r always equals
  that quantity). But `r` is *not* constant in `d_a`: `w` changes because the
  window slides over the fixed interferer points `ψ_ij` and `pd_i(ψ)` grows with
  `ψ`. So `d_a` acts mainly through window selection, not through `+d_a`.

### 4. Predictions before the experiments (from the mechanism)

- **Q1 (honest Σd_i ≤ D_flow):** the unconstrained optimum (~810–815) *over-spends*
  each flow's budget (λ>1, γ>0 ⇒ `Σ d_i > D_flow`). Under the budget constraint
  the interferer channel can still be exploited by *reshaping* (make some tasks
  later at the cost of making others earlier), but earlier tasks are more
  aggressive interferers, so I expect a **partial** recovery — clearly above
  `pd` (which is budget-feasible) but **below** ~810. Prediction: budget-feasible
  optimum somewhere in ~740–810, i.e. *partly* an artifact.
- **Q2 (order vs magnitude):** λ changes only magnitude at fixed EDF order and
  already buys ~+100; permuting a fixed deadline multiset changes only order.
  Prediction: **magnitude dominates**, order gives a smaller (but nonzero) gain;
  the extra systems concentrate at mid-high utilization (where interference
  matters most).
- **Q3 (adaptive):** with all flows identical in structure (3 tasks) the optimal
  γ should be roughly universal, but its value should track the load —
  **higher `u` ⇒ larger optimal γ** (more interference to suppress). A per-flow
  rule on structure should give little beyond a good global γ *on this pool*.
- **Q4 (closed form vs search):** an unconstrained search can rediscover the
  ~815 optimum but only by overspending; a **budget-constrained** search should
  beat every closed form and recover most of the "honest" part, at a large
  time cost. Prediction: search headroom over the best closed form exists but is
  modest; cost is orders of magnitude larger. *(Outcome: the first two parts held
  strongly; the "modest" headroom guess was **wrong** — the search added +80
  (unconstrained) and +127 (budget) over the closed forms. See Q4.)*
- **Q5 (generalisation):** if the interior optimum is a genuine mechanism effect
  (interference suppression vs the limit/mis-alignment penalty) it should
  **survive** across sizes/mappings/deadline factors; if it is a quirk of this
  single 5×3×3 pool it may shift or vanish. I will test this directly.

---

## Baseline

Reproduced with `python run.py --per-level -q` (seed 42, 50 systems × 20 levels
= 1000 instances); output saved in `tmp/baseline.txt`.

- pd:            **708** / 1000
- pd_global:     **744** / 1000
- (reference) equal = 407, period_prop = 320 — both reproduced.
- AGENTS.md γ table (708/773/804/813/813/**815**/803/786/773/758/744) and λ
  table (457/619/708/784/**810**/793/776) reproduced exactly
  (`tmp/repro.py`).

**Numerical caveat.** The analysis is sensitive at the schedulability
boundary. A budget-normalised γ family at `g=0` reproduces pd's deadlines only
up to one floating-point rounding (`scale = D/sum(shares) ≈ 1±1e-16`) yet scores
710 vs 708. Differences of ≤ ~4 instances between very close variants are
therefore within floating-point noise; every headline gap below is much larger.

## Q1 — artifact or real? (`Σ d_i ≤ D_flow`)

**Prediction (from the mechanism).** The unconstrained optima (γ>0, λ>1) all
*overspend* each flow's budget; the "larger deadlines" benefit is obtained
partly by inflating deadlines beyond `D_flow`. Under the honest constraint
`Σ_i d_i ≤ D_flow` the interferer channel can still be exploited by reshaping
(back-loading: later tasks bigger, earlier tasks smaller), but earlier tasks are
then more aggressive interferers, so only a **partial** recovery is expected —
clearly above pd, below ~810.

**What I ran.** Budget-feasible families (`tmp/q1_budget.py`, `tmp/q1b_budget.py`),
each verified to satisfy `Σ d_i = D_flow` exactly per flow:

- `gamnorm(g)`: cumulative shape `d_i = (share_i + g·prefix_{i-1})` rescaled so
  the flow sum is exactly `D_flow`.
- `back_load(β)`: weights `w_k ∝ β^k`, normalised — arbitrary back-loading.
- plus `pd`, `pd_global`, `gamma0.5`, `lambda2` for contrast.

Budget check (system 0, u=0.75): `pd`, `gamnorm`, `back_load` give
`sum(d_i)/D_flow = [1.0,1.0,1.0,1.0,1.0]`; `gamma0.5` gives
`[1.46,1.56,1.79,1.88,1.61]`; `pd_global` gives `[1.92,2.12,2.58,2.75,2.22]`.

**Results (seed 42, 50×20 = 1000).**

| rule | budget | schedulable |
|---|---|---|
| pd | Σ=D_flow | 708 |
| gamnorm g=0.5 | Σ=D_flow | 760 |
| gamnorm g=0.6 | Σ=D_flow | 762 |
| back_load β=1.5 | Σ=D_flow | 529 |
| **search, budget-feasible** | Σ=D_flow | **889** |
| pd_global | ×1.9–2.75 | 744 |
| γ=0.5 | ×1.5–1.9 | 815 |
| λ=2 | ×2 | 810 |
| closed best (g=0.3,λ=2) | ×1.3–1.9 | 865 |
| **search, unconstrained** | ×(free) | **945** |

**Conclusion (revised by Q4).** If one only optimises the *single-parameter*
γ/λ families under the budget, the best is ≈762 — which suggested "half the
gain is overspending". But that comparison was confounded: γ and λ conflate
shape and magnitude. When the *shape* is optimised **within the budget** by the
local search of Q4, the budget-feasible optimum is **889/1000** — *higher than
every unconstrained closed form* (best 865). The unconstrained search reaches
only **945**, so the entire "overspending" advantage, once shape is optimised on
both sides, is ≈945−889 = **56 instances (~6%)**, and **none** of the single-
parameter gain was *necessary*: a well-shaped budget-feasible assignment beats
γ=0.5 (815) and λ=2 (810) by a wide margin. In other words:

> the "larger deadlines help" effect is **real** (a property of the interference
> bound), not primarily an overspending artifact. What *is* an artifact is the
> belief that you must overspend: the same effect is reachable by reshaping the
> deadlines within `Σ d_i ≤ D_flow`. The earlier "≈half artifact" reading was an
> artifact of comparing the budget-feasible closed-form family (`gamnorm`)
> against the overspending closed-form family (γ/λ), each of which is a thin
> slice of the true 2-D landscape.

Caveats: (i) 889 is a *lower bound* — the search uses only 1 pass and 3
candidate multipliers, so the true budget-feasible optimum is ≥889;
(ii) the constraint is enforced exactly (`tmp/check_budget.py`: 0 violations of
`Σ d_i ≤ D_flow` and of `d_i ≤ D_flow`, 15 instances × 5 flows); (iii) fp noise
is ±~4, negligible at these gaps.


## Q2 — order vs magnitude

**Prediction.** λ isolates magnitude (order fixed); permuting a fixed deadline
multiset isolates order. From the mechanism I expected magnitude to dominate and
any extra systems to appear at mid–high utilization.

**What I ran.**

1. *Pure order* (multiset and sum fixed): all 6 permutations of each flow's pd
   shares (`tmp/q2_order.py`).
2. *Magnitude at fixed order*: λ·pd.
3. *Shape at fixed budget*: `gamnorm(g)`.
4. *Shape × magnitude 2-D grid* (`tmp/q2b_shape.py`, `tmp/q2c_grid.py`):
   budget-normalised shape `gamnorm(g)` then scaled by λ.

**Results.**

- **Pure reordering never helps.** Of the 6 permutations of the pd share
  multiset, the identity (pd) is best at **708**; every other permutation is far
  worse: reverse (2,1,0)=240, (0,2,1)=182, (1,0,2)=209, (1,2,0)=70, (2,0,1)=70.
  The same holds for the *best* budget shape: permuting `gn0.6` gives
  762/331/305/152/89/175 (identity best) and permuting the cumulative shape gives
  744 for the identity but 135–379 otherwise (`tmp/q2d_perm_shape.txt`). So the
  gain is **not** a permutation of a fixed multiset; it changes the multiset
  itself (later tasks get absolutely larger deadlines) plus a global scale.
- **Magnitude alone** (λ·pd): 708 → 811 at λ=2.0 (+103).
- **Shape alone at fixed budget** (`gamnorm`): 708 → 762 (+54).
- **Shape × magnitude** (the actual source of the γ benefit): the 2-D grid
  reaches **865** (`g=0.3, λ=2.0`; `tmp/q2c_grid.txt`), well above either
  one-dimensional optimum. γ=0.5 (815) is a *single point* in this 2-D family and
  is not the best available. The schedulable count over the (g,λ) grid
  (rows g, cols λ; `tmp/q2c_grid.txt`):

  | g\λ | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 4.0 |
  |---|---|---|---|---|---|---|
  | 0.0 | 710 | 787 | **811** | 795 | 795 | 780 |
  | 0.2 | 751 | 847 | **864** | 862 | 851 | 848 |
  | 0.3 | 748 | 845 | **865** | 859 | 858 | 854 |
  | 0.5 | 760 | 845 | 857 | **865** | 862 | 843 |
  | 0.6 | 762 | 857 | 864 | 853 | 852 | 836 |
  | 1.0 | 728 | 797 | 805 | 804 | 799 | 788 |

  The optimum is a broad plateau (~855–865) for g∈[0.2,0.6], λ∈[2,2.75]; the
  single-parameter families only sample its edges.

**Conclusion.** The gain is **not** an ordering/permutation effect at all; it is
(a) a specific **back-loaded shape** (later tasks in a flow get relatively
larger deadlines) plus (b) a **magnitude** inflation. At a fixed budget the shape
alone is worth ~+54; magnitude on the pd shape is worth ~+103; and combining
them (865) shows the two interact. Where the extra systems come from
(per-level, from `tmp/q1b_budget.py`): pd→γ0.5 gains appear only for u≳0.65 and
grow toward u=0.9 (e.g. u=0.837: 15→24; u=0.900: 6→14), i.e. exactly where
interference is the binding constraint. **The 2-D closed form is still not
optimal**: the local search of Q4 reaches 945 (unconstrained) and 889
(budget-feasible), so shape+magnitude is the right *decomposition* but not the
full optimum.

## Q3 — adaptive γ/λ per flow

**Prediction.** All flows share the same 3-task structure and every processor is
scaled to the *same* utilization by `set_utilization`, so there is little
structure for a per-flow rule to exploit. I expect the optimal (g,λ) to be
essentially universal, with **no meaningful gain** from adapting; a purely
load-driven λ might track u weakly.

**What I ran** (`tmp/q3_adaptive.py`): a per-level (g,λ) grid (g∈{0,0.2,0.4,0.6},
λ∈{1.0,…,3.0}), a smooth load-adaptive rule `λ(u)=a+b·u` at fixed g, and a
per-flow rule where `λ_f` is interpolated by the flow's own deadline factor
`r_f = D_f/(n·T_f) ∈ [0.5,1]`.

**Results (seed 42, 50×20 = 1000).**

- The per-level grid optimum is **flat and noisy**: for u≤0.65 almost every
  (g,λ) gives 50/50 (no signal); for u≥0.8 the optimum is a broad plateau,
  g∈[0.2,0.4], λ∈[1.75,2.5]. The "best per level" table is therefore
  overfit — several near-ties.
- Best global constant from this grid: g=0.2, λ=2.0 → **864**.
- Load-adaptive: `λ(u)=2.8u` (g=0.3) → **864** — *identical* to the global
  best, not better. Other slopes: `0.5+2.2u`→856, `2.5u`→848.
- Per-flow λ from the deadline factor: best `λ∈[2.0,2.5]` → **855**, *worse*
  than the global best (849 for its own g=0.4 baseline, 864/865 global).

**Conclusion.** On this pool the optimal shape/magnitude is **essentially
universal** (γ-family optimum γ*≈0.2–0.3 once magnitude is free, or γ*≈0.5 if
you force the unnormalised single-parameter family); a per-level or per-flow
adaptive rule gives **no improvement** over the best global constant, and
mis-specified adaptivity (wrong slope, wrong per-flow variable) *hurts*. The
prediction is confirmed: there is no exploitable per-flow structure in this
pool.

**Caveat tested (`tmp/q6_unequal.py`).** The pool equalises processor
utilizations, so I also built systems with *unequal* loads (proc0/1/2 scaled to
0.60/0.75/0.90) and compared a single global λ against a λ chosen per task from
its own processor's utilization. Global g=0.2, λ=2.0 → **44/50**; the best
per-processor rule → **43/50** (equal-load control: global → 46, per-processor →
45). So even with unequal loads, per-processor adaptation does **not** beat the
best global constant — adaptivity is not where the headroom is. (Honest
uncertainty remains: a *properly fitted* per-processor rule with more than one
parameter, or a joint shape+magnitude rule, was not exhaustively searched; I
only tested two λ-slopes at fixed g.)

## Q4 — closed form vs search

**Prediction.** An unconstrained search can beat the closed forms but may do so
by overspending; a **budget-constrained** search should beat the budget-feasible
closed form and recover most of the honest part. Headroom over the best closed
form should exist but be modest, at orders-of-magnitude higher cost.

**What I ran.** Greedy coordinate descent on each task's deadline, minimising
total tardiness `Σ_f max(0, wcrt_f − D_f)` (objective computed with
`limit_factor=3` so wild candidates abort cheaply; `algorithms/search.py`).
Initialised from the best closed form (`g=0.3, λ=2.0`) for the unconstrained
variant and from `gn0.6` for the budget variant. Two experiments:
`tmp/q4_compare.py` (population 10, 3 hard levels u∈{0.711,0.795,0.879} = 30
instances) and `tmp/q4_full.py` (full 1000 instances).

**Results (30 hard instances).**

| technique | schedulable | total technique time |
|---|---|---|
| pd | 15 | 0.000s |
| λ=2 | 16 | 0.000s |
| γ=0.5 | 18 | 0.000s |
| closed best (g=0.3, λ=2.0) | **19** | 0.000s |
| budget closed (`gn0.6`) | 16 | 0.000s |
| search, passes=1 | **26** | 11.05s |
| search, passes=2 | 26 | 15.75s |
| search, budget, passes=2 | **25** | 21.23s |

(total technique time is the search's *internal* analyses; closed forms do a
single assignment. Per hard instance the search does hundreds of analyses, i.e.
~0.4–0.7 s vs microseconds.)

**Results (full 1000 instances, `tmp/q4_full.py`).**

| technique | schedulable | total technique time |
|---|---|---|
| closed best (g=0.3, λ=2.0) | 865 | 0.005s |
| budget closed (`gn0.6`) | 762 | 0.005s |
| search, unconstrained (passes=1) | **945** | 149.6s |
| search, budget-feasible (passes=1) | **889** | 533.4s |

Per level (schedulables / 50): the search closes most of the gap everywhere and
is decisive at high u: closed best at u=0.900 → 20, budget search → 22,
unconstrained search → 33; at u=0.879 → 26 / 29 / 39.

**Conclusion.** Search finds **substantial** headroom over the best closed form
(865 → 945 unconstrained; 762 → 889 under the budget), so the closed-form
families were far from optimal — the 2-D (shape, magnitude) family is *not* the
whole story. The budget-feasible search even **beats every unconstrained closed
form**. Cost: the searches do hundreds of full analyses per instance, i.e.
~0.15 s (unconstrained) / ~0.53 s (budget) per instance versus ~5 µs for the
closed forms — roughly **10⁴–10⁵× slower** for the extra ~+80–127 instances. The
passes=1 search is a weak search (1 pass, 3 multipliers); more compute would
only raise these numbers, so 945/889 are **lower bounds**.


## Q5 — generalisation

**Prediction.** If the interior optimum is a genuine mechanism effect it should
survive other pool shapes, mappings and deadline factors; if it is a quirk of the
5×3×3 seed-42 pool it may shift or vanish.

**What I ran** (`tmp/q5_general.py`, `tmp/q5b_general.py`). Built fresh pools
directly with `pool.generate_system` (SIZES only contains 15) — 25 systems each,
10 utilization levels in [0.5,0.9], 250 instances per configuration — and swept
the γ-family (unnormalised) and the λ-family (budget-normalised). **Warning:
these use a different pool size and only 10 levels, so the counts are not
comparable to the 1000-instance tables**; they are compared within each row.

| configuration | pd(γ0/λ1) | best γ | best λ | interior optimum? |
|---|---|---|---|---|
| 5×3×3 balanced | 179 | γ0.3 → 219 | λ1.5 → 216 | yes |
| 4×3×2 | 200 | γ0.3 → 219 | λ1.5 → 216 | yes |
| 6×3×3 | 188 | γ0.3 → 208 | λ1.5 → 212 | yes |
| 4×4×3 | 160 | γ0.3 → 201 | λ2.0–2.5 → 200 | yes |
| 8×2×4 | 183 | γ0.3 → 193 | λ1.5 → 187 | yes |
| 3×5×3 | 125 | γ0.5 → 169 | λ3.0 → 174 | yes (peaks 3.0, then 170/170/169) |
| 5×3×3 random mapping | 179 | γ0.4 → 210 | λ2.0 → 205 | yes |
| 5×3×3 deadline factor 0.3–0.5 | 73 | γ0.2 → 77 | λ1.5 → 79 | weak/marginal |
| 5×3×3 deadline factor 0.8–1.0 | 212 | γ0.4 → 232 | λ2.0 → 228 | yes |
| 5×3×3 deadline factor 1.0–2.0 | 248 | — | — | saturated (~250) |

**Conclusion.** The interior optimum **survives** across chain lengths
(2–5 tasks/flow), flow counts, processor counts, the mapping (balanced vs
random) and reasonable deadline factors — so it is a genuine property of the
interference mechanism, not an artifact of the single seed-42 pool. Its
*location* shifts with structure: deeper flows want larger λ and γ (3×5×3 peaks
at λ≈3, γ≈0.5) because each task has more predecessor jitter/interference to
absorb. The gain shrinks to nearly nothing when deadlines are very tight
(0.3–0.5) or very generous (1.0–2.0), consistent with the mechanism (tight
budgets leave no room to reshape; generous budgets make everything schedulable).

## Reproduce

All scripts live in `tmp/`; run them from the sandbox root with the repo venv
(`../../../.venv/bin/python`, Python 3.13 — `python` resolves to it here).
Unless noted, pool = seed 42, size 15, population 50, 20 levels = 1000
instances.

| command | what it produces |
|---|---|
| `python run.py --per-level -q` | baseline + registered techniques (`tmp/final_run.txt`, `tmp/baseline.txt`) |
| `python tmp/probe_wi.py` | micro-probe of eq (1) and the `d_a` cancellation |
| `python tmp/repro.py` | AGENTS.md γ and λ tables + `equal`/`period_prop` |
| `python tmp/q1_budget.py`, `python tmp/q1b_budget.py` | Q1 budget-feasible families + budget verification + per-level |
| `python tmp/q2_order.py` | Q2 permutations (order) vs λ (magnitude) |
| `python tmp/q2b_shape.py`, `python tmp/q2c_grid.py` | Q2 shape×magnitude 2-D grid |
| `python tmp/q2d_perm_shape.py` | Q2 permutations of the best budget/unconstrained shapes |
| `python tmp/q3_adaptive.py` | Q3 per-level grid + adaptive rules |
| `python tmp/q6_unequal.py` | Q3 caveat: unequal per-processor loads, per-processor λ |
| `python tmp/q4_compare.py` | Q4 closed form vs search (population 10, 3 hard levels) |
| `python tmp/q4_full.py` | Q4 full-pool budget vs unconstrained search |
| `python tmp/check_budget.py` | verifies the budget search respects `Σd_i ≤ D_flow` and `d_i ≤ D_flow` |
| `python tmp/q5_general.py`, `python tmp/q5b_general.py` | Q5 other sizes/mappings/deadline factors |

Registered techniques: `algorithms/shape.py` (`pd_cum_scaled` = 865,
`pd_budget_cum` = 762) and `algorithms/adaptive.py` (`pd_adaptive_u` = 864).
The search is `algorithms/search.py` (deliberately unregistered: too slow for
`run.py`).

## Conclusions

**The mechanism (from reading the analysis).** The analysis is *roughly* a
local-EDF busy-window analysis in which the only place an interferer's local
deadline enters is eq (1), `_wi`, through `pd = floor((j_i + D − d_i)/T_i)+1`
(or 0 if `D < d_i`) capped by the number of releases. A **later interferer
deadline ⇒ smaller `pd` ⇒ fewer counted interfering activations ⇒ smaller
WCRTs**. The analysed task's own deadline enters eq (9) additively but *cancels
exactly* against the shifted `ψ_ab` window; it acts mainly by selecting which
interferer deadline points fall in the window (verified numerically in
`tmp/probe_wi.py`). Hence "larger local deadlines make more systems schedulable"
is a genuine, explainable property of this *approximate* interference bound.

**Answers.**

1. **Artifact or real?** Real. Restricting to `Σ d_i ≤ D_flow` (and `d_i ≤
   D_flow`), the best *closed form* drops to 762, but a budget-feasible *search*
   reaches **889** — *above every unconstrained closed form* (best 865). With
   both sides searched, overspending is worth only ~56 instances (~6%). The
   earlier "half artifact" impression came from comparing two thin
   single-parameter families; the effect is a property of the analysis, reachable
   without overspending by choosing the shape well.
2. **Order vs magnitude.** **Neither "order" nor "magnitude" alone**: permuting
   a fixed deadline multiset never helps (identity is optimal; all other
   permutations collapse to 70–379), so EDF order per se is not the lever.
   The gain decomposes into a **back-loaded shape** (later tasks relatively
   larger; +54 at fixed budget) and a **magnitude** scale (+103 on the pd
   shape); the 2-D combination reaches 865, and search beyond that 945. Gains
   appear only for u ≳ 0.65 and grow with u.
3. **Adaptive rule.** No gain. On this pool (equal processor utilization,
   uniform 3-task flows) the optimal (g, λ) is essentially universal
   (γ*≈0.2–0.3 once magnitude is free; γ*≈0.5 in the unnormalised single-
   parameter family). Load-adaptive λ(u) ties the global best (864 vs 865) and
   per-flow/per-processor rules are *worse*, even with unequal processor loads
   (44 vs 43). There is no exploitable secondary structure here.
4. **Closed form vs search.** Search adds a lot: 865 → 945 unconstrained,
   762 → 889 budget-feasible. Cost: ~0.15–0.53 s/instance (hundreds of analyses)
   versus ~5 µs — 10⁴–10⁵× slower. Both search numbers are lower bounds
   (1 pass, 3 multipliers).
5. **Generalisation.** The interior optimum **survives** across chain lengths
   (2–5 tasks/flow), flow/processor counts, balanced vs random mapping, and
   reasonable deadline factors; it is not a seed-42 artifact. Its location
   shifts with structure (deeper flows want larger γ and λ; 3×5×3 peaks near
   λ≈3). It weakens when deadlines are very tight (0.3–0.5) or very generous
   (1.0–2.0).

**Overall.** Larger local deadlines help *because the reference analysis credits
an interferer with fewer activations the later its deadline*; this is the whole
story and it explains every number above. The headline "~810/815" figures are
not the limit of the effect: a better-shaped, still budget-feasible rule reaches
889, and an unrestricted search reaches 945.

**What is proved vs conjectured.**

- *Proved (reproducible):* baseline counts; the γ/λ tables; the eq (1) direction
  and the `d_a` cancellation (`probe_wi.py`); budget compliance of the budget
  techniques; the count of every family and search above; the noneffect of
  permutation and of adaptivity on this pool; the generalisation table.
- *Conjectured / uncertain:* the search numbers are lower bounds (a stronger
  search goes higher) — the *ceiling* is not established; whether
  per-processor adaptivity could help under a richer parameterisation is not
  settled (only two λ-slopes were tried); and none of this claims the reference
  analysis is *accurate* — it is explicitly approximate, and the techniques
  optimise *it*, not the real system.

**Deliverables.** `algorithms/shape.py` (`pd_cum_scaled` = 865,
`pd_budget_cum` = 762), `algorithms/adaptive.py` (`pd_adaptive_u` = 864),
`algorithms/search.py` (search, unregistered), all probe/experiment scripts in
`tmp/` (see *Reproduce*).

