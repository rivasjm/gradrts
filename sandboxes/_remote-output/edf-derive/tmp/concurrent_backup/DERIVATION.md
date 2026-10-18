# Derivation

> Deliverable for `sandbox-edf-derive`. Goal: a **rule** `d = g(system)` that
> assigns local deadlines to maximise the number of schedulable instances of the
> frozen pool (`frozen/pool.py`, seed 42, 50 systems × 20 levels = 1000
> instances), judged by the holistic local-EDF analysis (`frozen/analysis.py`).
>
> **Reproducibility.** Everything below is produced with the repository venv,
> **Python 3.13** (`../../../.venv/bin/python`), on this sandbox's frozen core,
> seed 42, size 15 (5 flows × 3 tasks, 3 processors), population 50, 20
> utilization levels in [0.5, 0.9] → 1000 instances. The analysis is numerically
> sensitive at the boundary; another interpreter shifts counts by a few
> instances. **What is derived** is marked *[derived]*; **what is only located
> empirically** is marked *[observed]*.

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
| `j_i` | release jitter = `max` WCRT of the flow-predecessors (`0` for the first task) | model (`Task.jitter`) |

A flow *is schedulable* iff `flow.wcrt ≤ flow.deadline`; the system iff every
flow is. `flow.wcrt` is the WCRT of the flow's **last** task. So the local
deadline `d_i` is **not** a requirement — it is a scheduling parameter (under
local EDF it also fixes the priority order).

Throughout, `a` is the **analysed** task and `i` ranges over the **interferers**
(the other tasks on `a`'s processor). All local deadlines are `d`.

---

## 1. Understanding the analysis (Step 0)

`frozen/analysis.py` computes `task.wcrt` as the fixed point of a holistic
response-time analysis. Its structure, equation by equation, and **every place a
local deadline appears**:

### 1.1 Busy period (eq 5, `_busy_period`)

`L` is the least fixed point of

```
L = ceil(L/T_a)·C_a + Σ_{i≠a, same proc} ceil((L + j_i)/T_i)·C_i ,   L ≥ C_a.
```

**No local deadline appears here.** `L` depends only on `(C, T, j)`. Deadlines
influence it only indirectly, through the jitters `j_i` (which are WCRTs, which
do depend on deadlines). The `p`-loop below runs `p = 1 … ceil(L/T_a)`.

### 1.2 Interference term (eq 1, `_wi`) — the crux

```
pl = ceil((t + j_i)/T_i)                                     # releases in a window of length t
pd = 0                                       if D < d_i      # D = the analysed window endpoint (psi)
   = floor((j_i + D - d_i)/T_i) + 1           otherwise       # "deadline points" of i at or before D
m  = min(pl, pd)                                             # activations credited to i
w_i = m · C_i
```

This is the **EDF demand cap**: only the interfering activations whose absolute
deadline `(q-1)T_i - j_i + d_i` is `≤ D` count. Two consequences, both used
below:

* **Direction `[derived]`.** `d_i` enters only as `D - d_i`. A **later**
  interferer deadline ⇒ smaller `pd` ⇒ fewer credited activations ⇒ **less**
  interference. It is *weakly monotone decreasing* in `d_i`.
* **Cutoff `[derived]`.** If `d_i > D`, then `pd = 0` and that interferer
  contributes **nothing** at that window point. Saturation is the other side:
  while `pd ≥ pl` the cap does not bite and increasing `d_i` changes nothing.

**Probe (`tmp/mechanism.py`).** For an interferer `(C=3, T=10, j=0)` and
`t=5, D=10`: `_wi = 3` for `d_i ∈ {0,1,5,9,10}`, then `0` for `d_i ∈ {11,50}`.
Weakly decreasing, with the exact `D < d_i` cutoff.

> This single term is *the whole story* of "larger deadlines help": later local
> deadlines on an interferer weaken the bound. It is a property of the
> **approximate** bound, not of the real system's behaviour (see §6).

### 1.3 ψ sets (eq 4, 6) and the `p`-loop

* eq (4) `_build_set_psi_ij` — the interferers' deadline points
  `{(q-1)T_i - j_i + d_i} ∪ {d_i}`. These **slide with `d_i`**.
* eq (6) `_build_set_psi_ab` — the analysed task's deadline points
  `{(k-1)T_a + d_a}`. **Depend on `d_a`**.
  (Note: the comprehension variable `p` shadows the argument, so this set does
  not actually depend on the activation index.)
* For activation index `p`, the enumerated window is
  `[ (p-1)T_a + d_a , p·T_a + d_a )` — it **slides with `d_a`**.

### 1.4 Fixed point (eq 8) and response (eq 9)

```
w   = p·C_a + Σ_{i≠a} _wi(i, w, psi)          fixed point, from w = p·C_a
r   = w - psi + d_a + j_a
R_a = max over (p, psi in the window of p) of r
```

Enumerating where `d_a` appears:

1. **Additively `+ d_a`** in eq (9).
2. **In `psi_ab = (p-1)T_a + d_a`** and in the window `[(p-1)T_a + d_a, pT_a + d_a)`.
3. Indirectly, by selecting which interferer points fall in the window and by
   the value of `pd_i(psi)`.

For the candidate `psi = psi_ab = (p-1)T_a + d_a`, the shift and the additive
term **cancel exactly**:

```
r = w - ((p-1)T_a + d_a) + d_a + j_a = w - (p-1)T_a + j_a       (d_a cancels).
```

So `d_a` mainly acts through **which interferer points are in the window** and
through the `+d_a` push on those (`psi_ij`) candidates. For an interferer point
`psi`, `r = w(psi) - psi + d_a + j_a`: raising `d_a` raises this term directly,
which is the **cost** of a large local deadline.

### 1.5 Limit-factor abort

While scanning `psi`, the moment any provisional `r > D_f · limit_factor` the
analysis raises and (with `reset=True`) wipes all WCRTs → the system counts as
unschedulable. `harness.is_schedulable` uses `limit_factor = 1`, so the test is
exactly "max provisional `r ≤ D_f` for every task". A rule that pushes a task's
provisional response past `D_f` loses the **whole** system. Our techniques read
responses with a huge `limit_factor` in `tmp/`/internally, but the **judge**
always uses `1`.

### 1.6 Summary of the lever `[derived]`

| where | effect of increasing that deadline |
|---|---|
| interferer `d_i` (eq 1) | fewer credited activations ⇒ **less** interference on `a` |
| analysed `d_a` (eq 9, `psi_ij`) | `+d_a` ⇒ **more** response for `a` |
| analysed `d_a` (eq 9, `psi_ab`) | **cancels**; only re-selects window points |

Every task is both an analysed task and an interferer, so the rule trades the
second column against the first. This is exactly the observed interior optimum
(§2.3).

---

## 2. Restricted cases (derive the shape)

### 2.1 One processor, two tasks, no jitter `[derived]`

Take two independent single-task flows on one processor, `a(C_a,T_a,d_a)` and
`b(C_b,T_b,d_b)`, `j_a=j_b=0` (the pool's first tasks). `tmp/mechanism.py`, with
`C_a=2, C_b=3, T=10, D=100`:

| `d_a \ d_b` | 2 | 4 | 6 | 8 | 10 | 20 |
|---|---|---|---|---|---|---|
| **2** | (5,5) | (3,5) | (2,5) | (2,5) | (2,5) | (2,5) |
| **4** | (5,3) | (5,5) | (3,5) | (2,5) | (2,5) | (2,5) |
| **6** | (5,3) | (5,3) | (5,5) | (3,5) | (2,5) | (2,5) |
| **8** | (5,3) | (5,3) | (5,3) | (5,5) | (3,5) | (2,5) |
| **10** | (5,3) | (5,3) | (5,3) | (5,3) | (5,5) | (2,5) |
| **20** | (5,3) | (5,3) | (5,3) | (5,3) | (5,3) | (5,5) |

Entries are `(R_a, R_b)`. Reading it off the analysis:

* If `d_a < d_b`: at `psi = d_a`, `pd_b = 0`, so `w = C_a = 2`; the self point
  gives `R_a = 2`. The only other candidate is `psi = d_b`, where
  `w = C_a + C_b = 5` and `r = 5 - d_b + d_a`. Hence

  ```
  R_a = max( C_a , C_a + C_b - (d_b - d_a) ),        R_b = C_a + C_b = 5.
  ```

  So `R_a = C_a` for **free** once the gap `d_b - d_a ≥ C_b`; otherwise a
  residual `C_a + C_b - (d_b-d_a)` remains.
* If `d_a = d_b`: both deadlines count each other (`pd = 1`), both get
  `C_a + C_b = 5`.

**Two derived facts.**

1. **Ordering = priority.** The set of tasks that delay `a` at its self point is
   exactly those on the processor with `d_i ≤ d_a`. Smaller local deadline ⇒
   higher local-EDF priority ⇒ *not* delayed by later-deadline tasks. Equal
   deadlines make everyone delay everyone (the `(5,5)` diagonal).
2. **Magnitude = cap relief.** The interferer's *absolute* deadline controls the
   residual: pushing `d_b` up by 1 reduces `R_a` by 1 until saturation. This is
   the eq(1) cap in its simplest form.

### 2.2 Add a chain / jitter `[derived]`

Now `t0 → t1` across two processors, plus one interferer `u` sharing `t1`'s
processor (`tmp/mechanism.py`, `T=10`, `C_0=C_1=2`, `C_u=3`, `d_0=2, d_1=8`):

| `d_u` | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|
| `R_0` | 2 | 2 | 2 | 2 | 2 |
| **end-to-end `R_1`** | 7 | 7 | 5 | 4 | 4 |

Raising the interferer's deadline `d_u` lowers the **end-to-end** response
(7 → 4), i.e. the eq(1) mechanism survives holistic coupling. Note `j_1 = R_0`
is unchanged (the interferer is not a predecessor), so this is interference, not
jitter.

**Cost propagation (why back-load).** Consider a flow `0,1,…,m-1`. Raising
`d_k` raises `R_k` (the self point `w(d_k)` counts more interferers), and
`R_k = j_{k+1}` is the **release jitter of the successor**, so the increase
**compounds along the chain** up to the end-to-end response. Raising the *last*
task's `d_{m-1}` does **not** compound — it has no successor. Therefore the
marginal cost of a large local deadline is **smallest for the last task and
largest for the first**. The benefit (less interference on co-processor tasks)
is the same wherever it is placed. A cost-minimising allocation therefore
**back-loads**: later tasks in a flow get larger local deadlines. `[derived]`

### 2.3 Several flows per processor ⇒ shape + magnitude `[derived direction, observed constants]`

With many tasks per processor the interference sets overlap and no closed form
drops out. The two derived levers give a two-parameter family. Write the WCET
shares `s_i = C_i·D_f/ΣC` (so `Σ_i s_i = D_f`) and the prefix `P_{i-1}`:

```
shape:      r_i = s_i + g·P_{i-1}                 (g ≥ 0 back-loads later tasks)
magnitude:  d_i = r_i · λ · D_f / Σ_j r_j         (λ > 0; Σ d_i = λ·D_f)
```

* `g = 0, λ = 1` is exactly `pd` (708); `g = 1, λ = 1` is `pd_global` (744).
* The directed prediction is `g > 0` (back-load) and `λ > 1` (inflate to push
  interferer points out of windows).
* **The constants `g, λ` are located empirically** (a 2-D sweep over the pool;
  `tmp/sweep_final.py`). The optimum is a **broad plateau** `g ∈ [0.2, 0.6]`,
  `λ ≈ 2.0`; the single best point on the grid is `g=0.3, λ=2.0 → 865`.

The interior optimum in `λ` is exactly the §1.4 tension: below it the caps do
not bite; above it the analysed task's `+d_a` push and the cap stop paying. The
uniform-`λ` sweep of the shipped `pd` shows the same interior optimum
(`tmp/eval_families.py`): λ = 0.5/0.75/1.0/1.5/2.0/3.0/5.0 →
457/618/**708**/787/**810**/793/775. (The interpolated γ family
pd→pd_global peaks at ~814 around γ≈0.3–0.5; `equal`/`allD` give 314/407.)

---

## 3. The rule

The rule has **two stages**: a one-pass closed form (the derived shape), then a
small fixed number of **bounded correction passes** that fix the magnitude
flow-by-flow. It is a single fixed function of the system: the same code runs
for every instance.

### 3.1 Seed — the one-pass closed form (`algorithms/bl_scale.py`)

For every flow, with `s_i = C_i·D_f/ΣC` and `P_{i-1} = Σ_{k<i} s_k`:

```
r_i = s_i + g·P_{i-1}
d_i = r_i · λ·D_f / Σ_j r_j
```

with **g = 0.3, λ = 2.0**. One pass, no analysis, no search. `[derived shape;
observed constants]`

### 3.2 Correction passes (`algorithms/bl_adaptive.py`)

The seed uses **one** magnitude for the whole system, but the pool's flows differ
(hardware mapping, deadline factor, wcet spread), so for some flows the seed is
too generous and for others too tight — precisely the residual the search
headroom exposes (§4). Correct it per flow. Repeat `P = 2` times:

```
analyse the system with limit_factor = ∞        # read untruncated responses R_f
for each flow f:
    φ_f = clip( (D_f / R_f)^α , [lo, hi] )
    scale every local deadline of f by φ_f
```

with **α = 0.6, [lo, hi] = [0.75, 1.25]**.

Rationale `[derived]`:

* `D_f < R_f` (flow late) ⇒ `φ_f < 1`: its local deadlines are too large for its
  own response; **shrink them**. This lowers its own response (the `+d_a` and cap
  terms).
* `D_f > R_f` (flow has slack) ⇒ `φ_f > 1`: it can afford **larger** deadlines,
  which suppress the interference it imposes on its processor-mates (eq 1).
* `α < 1` **damps** the step (the response is not linear in the deadlines) and
  the **clip is a trust region**: (i) an aggressive step overshoots, and (ii)
  shrinking a flow's deadlines *raises* interference on its processor-mates, so
  a big downward step has collateral cost. The clip keeps each pass local.

The corrected deadlines need **not** sum to `D_f` (neither does the seed).

### 3.3 Statement, assumptions, complexity

**Assumptions.** (A1) The pool's structure — chains mapped round-robin, local
EDF on coupled processors — with the frozen holistic analysis as the judge.
(A2) Some slack may be "spent" beyond the flow budget: `Σ_i d_i > D_f` is
allowed, because the local deadline is not a requirement (we verified this is an
intrinsic property of the bound, §6). (A3) The correction passes read the
system's own response; that is permitted ("it may read the system") and is a
fixed number of passes, not a search.

**Cost.** `bl_scale`: one pass, 0 analyses. `bl_adaptive`: `2` analyses of the
full system. Both are dominated by the judge's own analysis; mean technique time
per instance ≈ 0.019 s (`bl_adaptive`) / ~1e-5 s (`bl_scale`).

**Non-iterative?** Yes, in the intended sense: a closed form plus a **fixed
count** (2) of correction passes with no trial-and-error over candidates.

---

## 4. Validation

All on the frozen pool: seed 42, size 15, population 50, 20 levels → **1000
instances**, Python 3.13 (repo venv). Exact command:

```
../../../.venv/bin/python run.py --per-level -q
```

| technique | schedulable | note |
|---|---|---|
| `pd` (shipped) | **708** | baseline |
| `pd_global` (shipped) | **744** | cumulative, budget-feasible |
| best γ family (reproduced) | ~814 | partial globalisation |
| best λ family (reproduced) | ~810 | uniform scale of `pd` |
| **`bl_scale`** (seed only, one pass) | **865** | `g=0.3, λ=2.0` |
| **`bl_adaptive`** (this rule) | **912** | seed + 2 correction passes |
| coordinate-descent headroom | **945** | own optimiser (`tmp/headroom.py`), lower bound |

The `pd`/`pd_global` counts reproduce the shipped values exactly. `bl_scale`'s
865 matches the best point of the (shape, magnitude) family in the 2-D sweep
(`tmp/sweep_final.py`: `g0.3_l2.0 = 865`; neighbours `g0.2_l2.0`,`g0.6_l2.0`,
`g0.3_l2.25` all 864). The headroom 945 is from a 1-pass coordinate descent
(init `g=0.3, λ=2.0`, multipliers 0.5–2.0, tardiness objective), so it is a
**lower bound** on the per-system optimum; the AGENTS-quoted reference is ~898.

> **Workspace note (not part of this derivation).** `algorithms/wad.py` also
> exists in this directory — a one-pass "work-and-position" rule
> (`d_i = D_f(Q·C_i/ΣC + S·i/(n-1))`, `Q=1.6, S=0.4`) whose module docstring
> refers to an earlier `DERIVATION.md`. It is **not** authored by, nor derived
> in, the present document; it was already present and is left untouched. For
> completeness it scores `wad` = 875 and `wad_lin` = 868 (its closest sibling),
> both below `bl_adaptive`'s 912. The rule delivered here is `bl_adaptive` (and
> its seed `bl_scale`).

**Per-utilization breakdown** (`run.py --per-level -q`):

| u | pd | pd_global | bl_scale | **bl_adaptive** |
|---|---|---|---|---|
| 0.500 | 50 | 50 | 50 | 50 |
| 0.521 | 50 | 50 | 50 | 50 |
| 0.542 | 50 | 50 | 50 | 50 |
| 0.563 | 50 | 50 | 50 | 50 |
| 0.584 | 47 | 50 | 50 | 50 |
| 0.605 | 47 | 49 | 50 | 50 |
| 0.626 | 47 | 48 | 49 | 50 |
| 0.647 | 47 | 48 | 48 | 50 |
| 0.668 | 46 | 46 | 48 | 50 |
| 0.689 | 44 | 44 | 48 | 49 |
| 0.711 | 39 | 43 | 48 | 48 |
| 0.732 | 39 | 42 | 47 | 47 |
| 0.753 | 33 | 37 | 45 | 47 |
| 0.774 | 31 | 32 | 43 | 45 |
| 0.795 | 27 | 28 | 42 | 44 |
| 0.816 | 20 | 23 | 38 | 42 |
| 0.837 | 15 | 19 | 33 | 41 |
| 0.858 | 11 | 17 | 30 | 37 |
| 0.879 | 9 | 11 | 26 | 35 |
| 0.900 | 6 | 7 | 20 | 27 |
| **Σ** | **708** | **744** | **865** | **912** |

Gains appear where interference binds (u ≳ 0.65) and grow with load; at u = 0.9
the rule makes 27/50 schedulable versus 6 for `pd`.

**Seed robustness** (`bl_adaptive`, other pool seeds — the metric pool is seed
42):

| seed | 1 | 7 | 13 | 42 | 99 |
|---|---|---|---|---|---|
| `bl_adaptive` /1000 | 913 | 903 | 922 | 912 | 898 |
| `bl_scale` /1000 | 856 | 820 | 847 | 865 | 824 |

So the rule is not a seed-42 fit: `bl_adaptive` stays ~900–922 across seeds, and
always beats `bl_scale` except when the seed is already easy (seed 42).

---

## 5. Validity domain and failure cases

**Where it is near-optimal.** At low load (u ≤ 0.6) everything is schedulable,
so no rule can help. In the mid/high range the correction passes close most of
the gap to the search: at u = 0.879 the rule gets 35 vs the headroom's 39
(§4); the residual gap to 945 is concentrated at u ≥ 0.84. There the analysis's
binding structure is per-task/erratic (the search finds wildly different
per-flow deadlines), which no flow-level correction can capture.

**Where the correction can fail / notes.**

* The correction is **flow-level**, so it cannot fix an intra-flow imbalance
  (e.g. one task badly placed while the flow-level response is acceptable). On
  the pool, a per-task version of the correction was **worse**
  (`tmp/adaptive2.py`: per-task 850 vs per-flow 897 at the same settings).
* A **too-wide** trust region is destructive: `[0.4, 4.0]` collapses to 719–888
  depending on `α`, versus 897–912 for `[0.75, 1.25]`. The response is strongly
  non-linear in the deadlines, and a big downward step on one flow raises
  interference on others.
* `α` too large over-corrects (`α = 1` with a tight clip still scores 888); `α`
  too small under-corrects. The plateau `α ∈ [0.5, 0.75]` is broad.
* More passes do not help: `P = 2` and `P = 3` give the same 912; `P = 4` gives
  910. The rule converges in two passes.

**Assumption checks.** (A2) is genuinely load-bearing: the seed overspends each
flow budget (`Σ d_i ≈ 2.0·D_f`, more after upward corrections). The
budget-feasible variant of the family is much weaker (`pd_global` 744, and the
budget-normalised shape at `λ=1` reaches ~762), so part of the gain comes from
spending beyond `D_f`. This is a property of the **bound**, not of the real
system (§6), and the pool imposes no budget constraint — so it is legitimate
under the stated objective but must be flagged.

---

## 6. Discussion

**Is this exploiting the approximate analysis?** Yes, deliberately and openly.
The analysis is stated to be approximate. Its interference term credits an
interferer with fewer activations the later its local deadline (eq 1); the rule
lifts local deadlines to exploit exactly that. In a real system, whether a task
is preempted by another does not depend on the second task's *local* deadline in
this simple way (the local deadline only orders the *local* EDF queue). So the
rule optimises the **reference bound**, which is the stated objective, and its
effect is a property of the bound. The honest summary: the rule is a
*justifiable* response to the given judge, not a claim about real hardware.

**What is proved vs observed.**

* *Derived:* §1.2 (direction/cutoff of eq 1), §1.4 (`d_a` cancels at `psi_ab`),
  §2.1 (two-task table and the residual formula), §2.2 (cost propagation ⇒
  back-load), §3.2 (the direction and need for damping/trust region of the
  correction), §2.3 (interior optimum in λ as the cost/benefit tension).
* *Observed / located:* the constants `g=0.3, λ=2.0, α=0.6, [0.75,1.25]`,
  `P=2`; the exact counts; the headroom 945 (a lower bound). The constants sit on
  broad plateaus and the rule is seed-robust, which is evidence they are not a
  fit to seed 42 — but they are not derived in closed form.

**Generalisation.** The direction (back-load + inflate, then correct per flow)
is derived from the analysis and should carry to other sizes/mappings/deadline
factors; only the constants should shift. This sandbox exposes only `size = 15`,
so a full cross-size study is out of scope here; the seed-robustness table is
the available evidence. **Open questions:** (i) can the correction be replaced
by a closed form for `λ` derived from the busy period? (ii) a properly-derived
trust region instead of `[0.75, 1.25]`; (iii) the residual gap at u ≥ 0.84 looks
inherent to flow-level correction.

**Reproduce.**

| command | output |
|---|---|
| `../../../.venv/bin/python run.py --per-level -q` | §4 table + per-level |
| `../../../.venv/bin/python tmp/explore.py` | §2.1 two-task sweep (seed probe) |
| `../../../.venv/bin/python tmp/mechanism.py` | §1.2 eq(1) direction, §2.1 table, §2.2 chain |
| `../../../.venv/bin/python tmp/eval_families.py` | γ/λ families, `equal`/`allD` |
| `../../../.venv/bin/python tmp/sweep_final.py` | (g,λ) plateau |
| `../../../.venv/bin/python tmp/adaptive.py`, `tmp/adaptive2.py`, `tmp/refine2.py` | correction-pass sweeps, per-task vs per-flow |
| `../../../.venv/bin/python tmp/headroom.py` | coordinate-descent headroom |

Registered rules: `algorithms/bl_scale.py` (`bl_scale`, 865) and
`algorithms/bl_adaptive.py` (`bl_adaptive`, 912). Scratch scripts in `tmp/`.
