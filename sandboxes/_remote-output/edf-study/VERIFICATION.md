# Independent verification of `NOTES.md`

**Verifier role.** This report tries to *falsify* `NOTES.md`. It was produced in a
separate session, reading `AGENTS.md`, `frozen/analysis.py`, `frozen/model.py`,
`frozen/harness.py`, `frozen/pool.py`, every module in `algorithms/`, every probe
script in `tmp/`, and re-running the experiments from scratch. `NOTES.md` was
**not** edited.

**Environment.** Repo venv, `../../../.venv/bin/python` = **Python 3.13.15**,
run from the sandbox root. Pool: `seed=42, size=15, population=50`, 20 levels in
[0.5, 0.9] → 1000 instances, unless a row states otherwise. Every number below
was regenerated with the venv unless prefixed otherwise.

**Commands used** (each `python` = the venv interpreter):

```
python run.py --per-level -q
python tmp/repro.py                 tmp/probe_wi.py
python tmp/q1_budget.py             tmp/q1b_budget.py
python tmp/q2_order.py              tmp/q2d_perm_shape.py
python tmp/q2c_grid.py              tmp/q3_adaptive.py
python tmp/q4_compare.py            tmp/q4_full.py
python tmp/q5_general.py            tmp/q5b_general.py
python tmp/q6_unequal.py            tmp/check_budget.py
# verifier-only probes (written for this pass):
python tmp/verify_edge.py           python tmp/verify_confirm.py
```

Verifier outputs are saved as `tmp/verify_*.txt` so they can be diffed against
the author's `tmp/*.txt`.

---

## 0. Verdict at a glance

| # | Claim (source) | Reproduced? | Verdict |
|---|---|---|---|
| 0.1 | Interpreter sensitivity: sys py3.9 shifts counts | yes (pd 708→713) | **confirmed** |
| 0.2 | Baseline pd=708, pd_global=744 | exact | **confirmed** |
| 0.3 | `equal`=407 / `period_prop`=320 | exact | **confirmed** (but label ≠ AGENTS.md) |
| 0.4 | AGENTS γ-table and λ-table | exact | **confirmed** |
| 0.5 | Registered techniques 865/762/864 | exact | **confirmed** |
| 1.1 | eq(1): later interferer deadline ⇒ fewer activations | exact | **confirmed** |
| 1.2 | `psi_ab` comprehension shadows `p` | code read | **confirmed** |
| 1.3 | `d_a` cancels in eq(9) for `psi_ab` | exact | **confirmed** |
| 1.4 | interferer channel: WCRT 168.2→92.1, non-monotone | exact | **confirmed** |
| 1.5 | `limit_factor=1` ⇔ "some provisional r > D_flow" | code read | **confirmed** |
| 2.1 | Q1 gamnorm/best budget = 760/762, back_load = 529 | exact | **confirmed** |
| 2.2 | Q1 budget search 889, unconstrained 945 | exact | **confirmed** |
| 2.3 | Q1 budget compliance (0 violations) | exact (+100 inst.) | **confirmed** |
| 2.4 | Q1 fp caveat: 708 vs 710, diff ~1e-16 | exact (2.5e-16) | **confirmed** |
| 2.5 | **Q1 table: closed best `×1.3–1.9`** | actual is `×2.00` | **REFUTED** (isolated) |
| 3.1 | Q2 permutations: identity best (708), rest 70–240 | exact | **confirmed** |
| 3.2 | Q2 magnitude (λ=2) → 810/811; shape → 762 | exact | **confirmed** |
| 3.3 | Q2 2-D grid, best 865 (`g0.3,l2.0`) | exact | **confirmed** |
| 4.1 | Q3 adaptive rules give no gain (864/855 …) | exact | **confirmed (as tested)** |
| 5.1 | Q4 30-instance table | exact | **confirmed** |
| 5.2 | Q4 full search 945 / 889 | exact | **confirmed** |
| 6.1 | **Q5 row "5×3×3 balanced" = γ219/λ216** | true values γ201/λ200 | **REFUTED** (row mis-copied) |
| 6.2 | Q5 other rows (sizes, mapping, deadlines) | exact | **confirmed** |
| 7.1 | "this is the whole story … explains every number" | — | **overstated** (rhetoric) |

Net: the study is **numerically reproducible and the central mechanism holds**.
Two concrete factual errors were found (Q1 overspend ratio; the Q5 "5×3×3" row),
plus a few minor imprecisions. Neither error overturns a conclusion.

---

## 1. Environment and baseline

### 0.1 Interpreter sensitivity — **confirmed**
AGENTS.md warns a different interpreter shifts counts. With the *system*
`/usr/bin/python3` (3.9.6) and only `pd`/`pd_global`:
`pd=713, pd_global=745` vs `pd=708, pd_global=744` on 3.13.15 → the warning is
real, and all reproduction below correctly uses the venv. (Verifier output in the
shell log; reproducible with the same one-liner under `/usr/bin/python3`.)

### 0.2–0.3 Baseline — **confirmed**
`python tmp/repro.py`:

```
pd 708, pd_global 744, equal 407, period_prop 320
```

Matches NOTES. **Caveat (naming, not a numerical error):** `AGENTS.md` states
`equal`, `period_prop` and "every task = flow.deadline" give **314, 320, 407**.
`tmp/common.py`'s `equal` sets *every task's deadline = flow.deadline* → **407**;
the technique AGENTS.md calls `equal` is the *equal split* `d_i = D_flow/n`, which
the verifier measured as **314** exactly. So NOTES' "equal = 407" is right for its
own definition but will look wrong to a reader following AGENTS.md's labels. Worth
a one-line clarification; no conclusion depends on it.

### 0.4 AGENTS γ/λ tables — **confirmed, exact**
`python tmp/repro.py` reproduced both tables digit-for-digit:
γ: 708/773/804/813/813/**815**/803/786/773/758/744;
λ: 457/619/708/784/**810**/793/776.

### 0.5 Registered techniques — **confirmed, exact**
`python run.py --per-level -q` reproduced `tmp/final_run.txt` exactly:
`pd_cum_scaled=865`, `pd_budget_cum=762`, `pd_adaptive_u=864`, plus per-level
identical to the author's `tmp/final_run.txt`.

---

## 2. Mechanism claims (the "Analysis" section)

### 1.1–1.2 eq(1) and the `psi` sets — **confirmed**
`python tmp/probe_wi.py` reproduced the notes' numbers exactly:
`_wi = 30.1223` for `d_i∈{0,1,10,100}`, collapsing to `0` once `d_i>D`.
The verifier's extra fine sweep (`tmp/verify_edge.py`, section F) confirmed
`_wi` is **exactly non-increasing** in `d_i` (0 non-monotone steps over a dense
grid) and matches the closed form `min(pl, pd)` with
`pd = 0 if D<d_i else floor((j+D−d_i)/T)+1`. The threshold is strict: `D=200`,
`d_i=200` → interference kept; `d_i=201` → 0.

The claimed comprehension bug in `_build_set_psi_ab` (the argument `p` is shadowed
by the loop variable, so the returned set is independent of `p`) is present in
`frozen/analysis.py:95-98`. **Confirmed by code inspection.**

### 1.3 `d_a` cancellation — **confirmed, exact**
For `psi = (p−1)T_a + d_a`, the probe prints `r == w−(p−1)T_a+j` for every `d_a`
(30.122 / 181.285 / 571.860). The additive `+d_a` is exactly cancelled by the
shifted `psi_ab`; the residual dependence on `d_a` is through `w` (window
selection over fixed interferer points). The mechanism story is correct.

### 1.4 Interferer channel — **confirmed**
Sweeping one proc-0 task's deadline moved another proc-0 task's WCRT
`168.17 → 92.12`, **with a non-monotone bump** (`151.56` at an intermediate value
between `138.45` and `92.12`). The notes' "weakly monotone *interference*, but
non-monotone WCRT" framing matches the data.

### 1.5 `limit_factor` — **confirmed**
`harness.is_schedulable` uses `limit_factor=1, reset=True`; `_task_analysis`
raises as soon as a provisional `r > D_flow`, wiping all WCRTs. The notes'
equivalence "early abort ⇔ some provisional r > D_flow" is correct by inspection.

---

## 3. Q1 — artifact or real?

### 2.1–2.3 — **confirmed, exact**
- `tmp/q1_budget.py`: `gamnorm` best = **762** (g=0.6), `back_load(1.5)` = **529**,
  `gn0.0` = **710**, `pd` = **708**, `gamma0.5` = 815, `lambda2` = 810.
- `tmp/q1b_budget.py` overspend ratios on system 0, u=0.75:
  `gamma0.5=[1.461,1.562,1.788,1.876,1.611]`, `pd_global=[1.922,2.123,2.576,2.752,2.222]`
  — match NOTES' quoted `[1.46…1.88]` / `[1.92…2.75]`.
- `tmp/q4_full.py`: `search_unbounded=945`, `search_budget=889` — **exact**, and
  the full per-level vector matches `tmp/q4_full.txt` cell-for-cell.
- Budget compliance: `tmp/check_budget.py` → 0 violations (reproduced); the
  verifier extended this to **100 instances × 5 flows** (`tmp/verify_edge.py`, E)
  → **0** violations of both `Σ d_i ≤ D_flow` and `d_i ≤ D_flow`. Confirmed.

**So the headline "889 > every unconstrained closed form (865)" is real.**

### 2.4 Floating-point caveat — **confirmed**
`tmp/verify_edge.py` (A): the largest relative difference between `pd`'s deadlines
and the budget-normalised `g=0` vector is **2.457e-16** (only flow2's
`D/sum(shares)` differs from 1, by 2.2e-16), yet the counts differ 708 vs 710.
The notes' explanation is correct: the 2-instance gap is pure fp noise, so any
gap ≤ ~4 between near-identical variants is not signal.

### 2.5 ❌ The "closed best" overspend ratio is wrong — **REFUTED (isolated)**
NOTES Q1 table row: `closed best (g=0.3, λ=2) | ×1.3–1.9 | 865`. The `λ=2` row
correctly says `×2`. But `shape_scale` sets `scale = λ·D_flow/Σd_i`, so the
per-flow sum is **exactly `λ·D_flow`**; for `(g=0.3, λ=2.0)` it is **×2.00 for
every flow**. Verified two ways: the registered `algorithms.shape.pd_cum_scaled`
and a fresh reimplementation, on u∈{0.6,0.75,0.9} (`tmp/verify_confirm.py`).
`algorithms/shape.py`'s own docstring ("sum(d_i)/D_flow ~ 1.3-1.6") is likewise
wrong. **This is a documentation error only** — it does not affect any
schedulability count or conclusion. The numbers `865` and `762` are correct.

---

## 4. Q2 — order vs magnitude

### 3.1 Permutations — **confirmed, exact**
`tmp/q2_order.py`: identity `(0,1,2)=708`; others `182, 209, 70, 70, 240`.
`tmp/q2d_perm_shape.py`: `gn0.6` permutations `762/331/305/152/89/175`; cumulative
`744/379/375/217/135/179`. Exact match to NOTES.

### 3.2 Magnitude / shape decomposition — **confirmed**
λ=2.0 → 810 (grid row `g0.0,l2.0` = 811; both appear in NOTES and both are
correct in their respective runs). `gn0.6` (shape, budget) → 762. Decomposition
holds.

### 3.3 2-D grid — **confirmed, exact**
`tmp/q2c_grid.py` reproduced the whole 11×11 table; the six rows quoted in NOTES
(g=0.0/0.2/0.3/0.5/0.6/1.0) match exactly, best `g0.3_l2.0 → 865`.
`tmp/q2c_grid.txt` and `tmp/verify_q2c_grid.txt` are identical outside timing.

**Minor imprecision (not a falsification).** NOTES says the pd→γ0.5 gains "appear
only for u≳0.65". The q1b per-level vector shows the gain already starts at
`u=0.584` (47→50), i.e. below 0.65. The cited examples (0.837: 15→24; 0.900: 6→14)
are exact. So "only for u≳0.65" should read "≳0.58".

---

## 5. Q3 — adaptive rules

### 4.1 — **confirmed (for what was tested)**
`tmp/q3_adaptive.py` output is **byte-identical** (outside timing columns) to the
author's `tmp/q3_adaptive.txt` (verified with a normalized `diff`): best global
constant 864, load-adaptive `λ(u)=2.8u` → 864 (tie), other slopes 856/848,
per-flow rules 855/838/838/821. The q6 unequal-load caveat also reproduced
exactly (`global g0.2 l2.0 = 44/50`, best per-processor `= 43/50`; equal-load
control 46 vs 45).

**Verdict nuance.** "Adaptivity buys nothing" is confirmed *within the tested
rules*. The notes explicitly flag that a richer per-processor parameterisation was
not searched; the verifier agrees this remains **uncertain** in general, and the
notes do not overclaim it.

---

## 6. Q4 — closed form vs search

### 5.1–5.2 — **confirmed, exact**
`tmp/q4_compare.py` (30 hard instances): pd 15, λ2 16, γ0.5 18, closed 19,
budget-closed 16, search1 26, search2 26, budget-search2 25 — exact, per-level
too.
`tmp/q4_full.py` (1000 instances): closed 865, budget-closed 762,
**search_unbounded 945, search_budget 889** — exact, per-level exact. (Verifier
wall times were larger — 850 s vs 708 s — because the job ran alongside others;
counts are deterministic and matched.)

**Minor inaccuracy.** NOTES says "the search is `algorithms/search.py`". The Q4
scripts actually import `search_technique` from **`tmp/search.py`**. The verifier
diffed the two files: they are **logically identical** (only `_tardiness` vs
`tardiness`, `_set` vs `set_deadlines`, `_project` vs `project_budget`,
`make_search` vs `search_technique`); `algorithms/search.py` additionally exposes
`make_search`. No numerical impact, but the reproducibility pointer is imprecise.
Also, `tmp/q4_compare.py`'s own docstring says "6 hard levels … 60 instances" while
the code uses 3 levels / 30 instances; NOTES correctly reports 30.

**Lower-bound caveat is honest and correct** — both searches use `passes=1` with 3
multipliers, so 945/889 are floors.

---

## 7. Q5 — generalisation

### 6.1 ❌ The "5×3×3 balanced" row is mis-copied — **REFUTED (row)**
NOTES table row: `5×3×3 balanced | pd 179 | γ0.3 → 219 | λ1.5 → 216`.

The verifier recomputed the 5×3×3 curve with the *same* custom loop
(`tmp/verify_edge.py`, D; identical to `q5b_general.py`'s `shuffle=False`):

```
gamma : {g0.0:179, g0.2:200, g0.3:201, g0.4:198, g0.6:199}
lambda: {l1.0:179, l1.5:195, l2.0:200, l2.5:196, l3.0:195}
```

True best for 5×3×3 is **γ≈201** and **λ=200** — *not* 219/216. The values
**219 and 216 are exactly the `(4,3,2)` row's values** (`tmp/verify_confirm.py`
reproduced `(4,3,2) gamma {g0.0:199, g0.3:219, g0.5:218}` and `tmp/q5_general.txt`
gives `(4,3,2) lambda {l1.0:200, l1.5:216, ...}`). So the row is a duplication
error. The
**qualitative conclusion survives** (5×3×3 still has an interior optimum:
179 → 201), but the two cited numbers are wrong.

### 6.2 Other Q5 rows — **confirmed, exact**
Every other row of `tmp/q5_general.txt` / `tmp/q5b_general.txt` reproduced exactly:
sizes `(4,3,2)=219/216`, `(6,3,3)=208/212`, `(4,4,3)=201/200`, `(8,2,4)=193/187`,
`(3,5,3)=169/174`; random mapping `210/205`; deadline factors `0.3–0.5 → 77/79`,
`0.8–1.0 → 232/228`, `1.0–2.0 → ~250`; the `(3,5,3)` λ peak at `3.0→174`
followed by `170/170/169`.

**±1 fp-noise in the "pd" column.** The `(4,3,2)` and `(4,4,3)` rows quote
`pd = 200` and `160`, which are the **λ=1** values; the **γ=0** values are `199`
and `161` (the two formulations differ only by fp normalisation). The custom Q5
loop therefore carries ±1 noise on the baseline column; these single-instance
differences should not be read as signal. (Same class of noise as the notes' own
"708 vs 710" caveat.)

---

## 8. The overall narrative

The notes' central, falsifiable claim —

> larger local deadlines help *because* eq(1) credits an interferer with fewer
> activations the later its deadline (a property of the interference bound, not
> of real execution)

— is **confirmed from the analysis and numerically** (sections 2–3). It is a
genuine mechanism of this approximate analysis, and the budget-feasible search
result (889) shows the effect is not purely an overspending artifact.

**Overstatement to flag.** The closing sentence "this is the whole story and it
explains every number above" is rhetorical. The mechanism *direction* is proved;
the *location* of the interior optimum and the exact optimum counts are observed,
not derived (the notes elsewhere admit the search ceiling is not established).
The honest framing is "the direction follows from eq(1); the magnitudes are
measured."

**Self-imposed-constraint caveat.** `Σ d_i ≤ D_flow` is not a requirement of the
analysis (which only checks `flow.wcrt ≤ D_flow`); it is a design choice. The
889-vs-945 comparison is still the right way to isolate the overspending
contribution, but "honest budget" should be read as a chosen operating point, not
a ground truth. The notes state it as `Σ d_i ≤ D_flow` throughout, so this is a
framing note, not an error.

---

## 9. Claims the verifier could **not** check

- The true **ceiling** of the search (both 945 and 889 are lower bounds; no
  exhaustive/heavier search was run) — notes correctly label these lower bounds.
- Whether a **richer per-processor / joint shape-magnitude adaptive rule** could
  beat the global constant (only two λ-slopes were tried) — notes already mark
  this uncertain; the verifier concurs it is unresolved, and expensive to settle.
- **Per-instance** optimality of the identity permutation in Q2 (only the
  aggregate 708 was checked); the notes' claim is an aggregate claim, so this is
  not a gap in their claim.
- Exact reproduction of the author's **wall-clock** search timings
  (149.6 s / 533.4 s); the verifier saw 180.8 s / 637.4 s under CPU contention.
  Counts are deterministic and matched; timings are host-dependent.

---

## 10. Reproduction summary

Every load-bearing number in `NOTES.md` (baselines; the γ and λ tables; the
`probe_wi` mechanism numbers; Q1 families, ratios and the 889/945 searches;
Q2 permutations, decomposition and the 11×11 grid; the Q3 grid and adaptive
counts; the Q4 30-instance and 1000-instance tables; the Q5 size/mapping/deadline
tables; the q6 unequal-load table; budget compliance) was regenerated on the repo
venv and **matched**. Verifier outputs are in `tmp/verify_*.txt`.

Only two factual errors were found, both in stated *ratios/values* rather than in
computed counts:

1. **Q1 table**: `closed best (g=0.3, λ=2)` overspend is **×2.00**, not `×1.3–1.9`
   (and `algorithms/shape.py`'s docstring "~1.3-1.6" is likewise wrong).
2. **Q5 table**: the `5×3×3 balanced` row quotes the `(4,3,2)` optima
   (`γ0.3→219`, `λ1.5→216`); the true 5×3×3 optima are `γ≈201` and `λ=200`.

Plus minor imprecisions: the "gains appear only for u≳0.65" bound (really ≳0.58);
the `equal` label mismatch with AGENTS.md (407 vs 314 for two different
definitions); the "the search is `algorithms/search.py`" pointer (results came
from the functionally identical `tmp/search.py`); and ±1 fp noise in the Q5
baseline column. None of these changes a conclusion.
