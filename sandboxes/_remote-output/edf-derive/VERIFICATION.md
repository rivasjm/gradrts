# Independent verification

**Verifier role.** I did not author `DERIVATION.md` or any file under
`algorithms/`. I read `AGENTS.md`, `frozen/{analysis,instrumented,model,pool,harness}.py`,
`DERIVATION.md`, the registered modules in `algorithms/`, and the probes in `tmp/`
(ignoring `tmp/concurrent_backup/`, which is a stale/parallel artefact). I then
re-derived the analysis facts, reproduced the numbers from scratch, and hunted
for counterexamples. I use the repository venv (**Python 3.13.15**,
`../../../.venv/bin/python`), the frozen pool (seed 42, size 15, population 50,
20 levels = 1000 instances) and the shipped judge.

**Verdict in one line.** The *headline result reproduces exactly* and the core
analysis facts are correct, but several *secondary* claims in `DERIVATION.md` are
wrong or unsupported, and the metric is extremely sensitive to
floating-point/formulation order, so "exact" counts are only meaningful for the
shipped code.

---

## 1. Reproduction of the headline numbers

Command (repo root):

```
../../../.venv/bin/python run.py --per-level -q
```

I ran it and got (technique: schedulable / 1000):

| technique | DERIVATION.md | my run | match |
|---|---|---|---|
| `pd` | 708 | **708** | ✅ |
| `pd_global` | 744 | **744** | ✅ |
| `wad` | 875 | **875** | ✅ |
| `wad_adapt` | 930 | **930** | ✅ |
| `wad_lin` | 868 | **868** | ✅ |
| `bl_scale` | 865 | **865** | ✅ |
| `bl_adaptive` | 912 | **912** | ✅ |
| mean time `wad_adapt` | ~0.023 s | 0.02277 s | ✅ |

The **entire per-utilisation column for `wad_adapt`** in §4 is byte-for-byte
identical to my run
(`50×10, 49, 48, 48, 46, 45, 45, 44, 37, 36, 32`). No technique reported an
error. `pd`/`pd_global` also match the AGENTS-quoted 708/744. **Confirmed.**

The code in `algorithms/wad.py` and `algorithms/wad_adapt.py` implements exactly
the rule written in §3.1/§3.2 (`Q=1.6, S=0.4`; `α=0.5, [0.75,1.25], P=3`,
analysis read with `limit_factor=1e9`). **Confirmed.**

---

## 2. The analysis facts (Step 0 re-derivation)

### 2.1 Busy period has no local deadline (§1.1) — confirmed
`_busy_period` uses only `(C, T, j)`. Direct code inspection. No counterexample.

### 2.2 eq(1) direction and cutoff (§1.2) — confirmed
`tmp/wad_probe.py` panel B reproduces the doc: victim `a(C=2,T=10,d_a=2)` with
interferer `t(C=3,T=10)` has `R_a=5` for `d_t∈{0,1,2}` and `R_a=2` for
`d_t≥5`; `format_detail` shows `pd=0` / `capped` at the cutoff. I additionally
searched 6394 single-processor sweeps (`tmp/verify_mono.py`) for a case where
**increasing** an interferer's deadline *increased* a processor-mate's response:
**0 non-monotone increases found.** The pointwise claim (`pd` weakly decreasing
in `d_t`) is rigorous, and I found no global violation on small systems.
**Confirmed.**

### 2.3 `d_a` cancellation at `psi_ab` (§1.4) — true as algebra, wording overstated
The algebra `r = w − ((p−1)T_a + d_a) + d_a + j_a = w − (p−1)T_a + j_a` is a
trivial substitution. **But** the follow-up sentence "along `psi_ab` the value of
`d_a` is *irrelevant*" is too strong: at `psi_ab = (p−1)T_a + d_a`, the fixed
point `w` still depends on `d_a` through
`pd_t = floor((j_t + (p−1)T_a + d_a − d_t)/T_t) + 1`. The doc immediately
qualifies this ("…it acts only by … the value of `pd_t(psi)` there"), so the
substance is right but the word "irrelevant" should be dropped. No impact on the
rule (the rule never uses this cancellation). **Confirmed (with an overstated
phrase).**

### 2.4 Two-task closed form (§2.1) — confirmed
I tested the stated formula for all `d_a ≤ d_b` over `{0..24}²`, `C_a=2, C_b=3,
T=10` (`tmp/verify_repro.py`): `d_a<d_b: R_a = max(C_a, C_a+C_b−(d_b−d_a))`,
`d_a=d_b: R_a = C_a+C_b`. **0 mismatches over 325 pairs.** For `d_a>d_b` the
observed `R_a` is always `5 = C_a+C_b`, matching the panel. `probe_A` exactly
reproduces the table in §2.1. **Confirmed.**

### 2.5 Interior optimum in λ (§2.3) — confirmed against the shipped probe
`tmp/eval_families.py` reproduces §2.3 exactly:
`457 / 618 / 708 / 787 / 810 / 793 / 775`. `equal=314`, `allD=407`, and the
γ-family peaks at `814` (γ=0.3 and 0.5). **Confirmed** — *but see §5 below: a
mathematically identical reformulation of the same λ rule gives
`457/619/708/784/810/793/776`.*

### 2.6 Limit-factor abort (§1.6) — confirmed
`harness.is_schedulable` uses `limit_factor=1, reset=True`; any provisional
`r > flow.deadline` wipes all WCRTs. I also proved the equivalent criterion: at
`p=1, psi=psi_ab`, eq(9) gives `r = w + j_a ≥ j_a = R_predecessor`, so along a
chain `R_k` is non-decreasing and `flow.wcrt` (the last task) is the **maximum**
over the flow. Hence "intermediate task past `D_f`" ⟺ "flow past `D_f`", and the
headroom optimiser's last-task tardiness objective is equivalent to the judge
(see §3). **Confirmed.**

---

## 3. Headroom reference (§4)

`tmp/wad_headroom.py` (population 25, 500 instances) printed:

```
instances=500  wad_adapt=462  optimiser=487
```

exactly the `487/500 ⇒ ≈974/1000` claimed. Because `R_last` dominates the chain
(§2.6), an optimiser that drives the sum of last-task tardiness to 0 is counted
schedulable by the judge — so **487 is a genuine achievable count, i.e. a valid
lower bound**. **Confirmed.**

Caveat: the derivation reports its *own* headroom (974) while the co-located
`bl_adaptive` docstring and `tmp/headroom_full.txt` report **945**, and AGENTS
quotes **~898**. `DERIVATION.md` never reconciles these three numbers. The 974 is
the one it uses and the one I could reproduce; the 945 is from a different,
weaker optimiser (`tmp/headroom.py`, `CAP=4.0`). **Unreconciled but not wrong.**

---

## 4. Seed robustness (§4)

I reproduced the `wad` row directly:

```
seed  1: wad=862   seed 7: wad=855   seed 42: wad=875   seed 123: wad=868
```

exactly the §4 table (`862/855/875/868`). `wad_adapt` across the same seeds:
**927/919/930/928**, also exactly the §4 table (see addendum). Across seeds the
one-pass rule therefore varies 855–875 and the adaptive rule 919–930; the claim
"not a seed-42 fit" holds. **Confirmed.**

---

## 5. Claims that are REFUTED or NOT REPRODUCED

### 5.1 §2.3: "`g = 1, λ = 1` is `pd_global`" — **refuted**
With the stated family `r_i = s_i + g·P_{i−1}`, `d_i = r_i·λ·D_f/Σr_j`:
* `g=0, λ=1` is algebraically `pd`, but the code returns **710**, not 708.
* `g=1, λ=1` is **not** `pd_global` algebraically. `Σ_i r_i = Σ_i cum_i`, so
  `d_i = D_f·A_i/(A_0+A_1+A_2)` with `A_i = Σ_{k≤i}C_k`, whereas
  `pd_global_i = D_f·A_i/ΣC`; these agree only if `2C_0+C_1=0`. Numerically it
  scores **728**, not 744.

This is a descriptive remark about a background family, not the delivered rule,
but it is simply false as written (and appears in `tmp/concurrent_backup` too).
**Refuted.**

### 5.2 §2.3: position ramp "868–875 across seeds" — **overstated / not reproduced**
`tmp/batch7.py` (the cited probe) gives, across seeds 1/7/42/123:
`rank_Q1.4_S0.3 = 873/863/848/872`, `rank_Q1.5_S0.3 = 865/860/849/871`. Seed 7
drops to **848–851**, far below the claimed 868. The real across-seed range of
the position-ramp family is ≈**848–875**. Likewise §4's `wad` range is 855–875,
not 868–875. **Refuted as stated.**

### 5.3 §5: "budget-normalised shape at λ = 1 reaches only ~762" — **not reproduced**
Neither natural reading gives 762: the back-loaded family at `λ=1` gives
**748**, and the WAD shape renormalised so `Σd=D_f` gives **775**. (This figure
is also inherited from `tmp/concurrent_backup/DERIVATION.md`.) **Not reproduced /
uncertain.**

### 5.4 §2.2 cost-propagation gradient — **weakly supported, not confirmed**
The claim "the marginal cost of a large local deadline is largest for the first
task and smallest for the last" did not survive a direct probe. On a 3-stage
chain (`a0→a1→a2`, one interferer per processor, `d_u=3`), raising `d_a0` by one
unit took the end-to-end response from 12 to 13; raising `d_a2` by one unit did
the *same* (12→13). The first-task increase did propagate along the chain, but
the end cost equalled the last-task cost — no gradient. The empirical reason the
position ramp helps (S=0 collapses to a λ-scaled `pd`) is real, but the stated
*derivation* of back-loading is not demonstrated. **Uncertain.**

### 5.5 §4 labels `pd_global` "budget-feasible" — **refuted**
`pd_global` sets the last task's deadline to exactly `D_f`, but its sum is
`Σd/D_f = Σ_i cum_i·D_f/ΣC > D_f`; measured over the pool it is
**min 1.050, mean 2.027, max 2.986**. So `pd_global` is *not* budget-feasible
(only `pd` is: exactly `1.000`). This matters because §5 uses "the
budget-feasible variant … `pd_global` 744" as the evidence that the gain comes
from overspending; the properly budget-feasible comparison is `pd` (708). The
qualitative point survives (with `pd`), but the stated evidence is wrong.
**Refuted.**

### 5.6 §5 failure-mode evidence is borrowed from the other family — **mis-attributed**
The claims "per-task correction was worse than per-flow at the same settings" and
"a too-wide trust region `[0.4,4.0]` collapses the score" are supported by
`tmp/adaptive.py` / `tmp/adaptive2.py`, which are probes for the **`bl_*` family**
(`g=0.3, λ=2.0`), not for `wad`/`wad_adapt`. `tmp/wad_adapt.py` only sweeps
trust-region widths `(0.8,1.25)`, `(0.85,1.15)`, `(0.9,1.1)` and the `[0.4,4.0]`
value does not appear in any `wad` probe. The qualitative conclusion (narrow
trust region is safer) is plausible from the pass/α sweeps, but the specific
`[0.4,4.0]` and per-task numbers do not belong to the delivered rule. **Overstated
/ mis-attributed.**

---

## 6. Is the success "exploiting the approximate analysis"?

The doc states openly (§6) that the gain comes from eq(1)'s demand cap: a later
*local* deadline makes an interferer contribute fewer activations to the bound,
which is a property of the approximate analysis, not of the modelled hardware.
My probes confirm this is **the** mechanism (`probe_B`, `probe_D`, and the
monotonicity sweep), and confirm the honest framing. Two checks on how far the
exploit goes:

* **No `d_i < C_i`.** I checked every task of every instance
  (`tmp/verify_model.py`): `pd`, `pd_global`, `wad`, `wad_adapt`, `bl_scale`,
  `bl_adaptive` never assign a local deadline below the task's own WCET
  (`wad_adapt` minimum `d/C = 1.09`). So the rule does **not** exploit the model
  by declaring physically impossible deadlines. **Confirmed clean.**
* **Budget overspend (A2) is load-bearing and real.** The WAD seed spends
  exactly `Σd/D_f = 2.200` for **every** flow (algebraically `Q + S·(n−1)/2`
  `= 1.6 + 0.4·(2)/2` for `n=3`); after the correction passes `wad_adapt`
  overspends `0.93–4.30×D_f` (mean 3.95) because it scales flows *up* as well as
  down and never renormalises. The doc flags the seed overspend correctly (§3.3,
  §5). Under the stated objective (local deadlines are free parameters, no
  budget constraint) it is legitimate, but part of the score is bought by
  *overspending the flow budget*, which is a model artefact. **Confirmed and
  correctly flagged** (only the `pd_global`-as-budget-feasible wording is wrong,
  §5.5).

---

## 7. Reproducibility caveat found during verification (important)

The judge is sensitive to **floating-point rounding in the deadline formula**,
not just to the interpreter. Mathematically identical reformulations shift the
count by a few instances on the same interpreter and pool:

* `d = λ·C·D/ΣC` computed as `λ*(C*D/ΣC)` gives `l0.75 = 618, l1.5 = 787,
  l5 = 775` (the shipped probe, and what `DERIVATION.md` quotes);
  computed as `(λ*C)*D/ΣC` gives `619, 784, 776`.
* The `bl` shape with `g=0, λ=1` is mathematically `pd` but scores **710** vs
  `pd`'s **708**.

Consequence: the "exact" totals are implementation-order dependent. The numbers
in `DERIVATION.md` reproduce *for its shipped code*, which is the right
standard — but the ±few-instance boundary sensitivity should be stated as
affecting even a single-operator reordering, not only a different interpreter.

---

## 8. Claim-by-claim summary

| # | Claim | Reproduced? | Counterexample? | Verdict |
|---|---|---|---|---|
| 1 | `pd=708, pd_global=744` | yes | no | confirmed |
| 2 | `wad=875`, per-level | yes | no | confirmed |
| 3 | `wad_adapt=930`, per-level | yes | no | confirmed |
| 4 | `bl_scale=865, bl_adaptive=912` | yes | no | confirmed |
| 5 | busy period has no deadline (§1.1) | code | no | confirmed |
| 6 | eq(1) direction/cutoff (§1.2) | yes | none found | confirmed |
| 7 | `d_a` cancels at `psi_ab` (§1.4) | algebra | wording | confirmed (overstated) |
| 8 | two-task formula (§2.1) | 325/325 | no | confirmed |
| 9 | λ family 457/618/708/787/810/793/775 | yes (shipped form) | float order | confirmed |
| 10 | `equal=314, allD=407`, γ peak 814 | yes | no | confirmed |
| 11 | abort = per-task `r≤D_f` (§1.6) | code + proof | no | confirmed |
| 12 | headroom 487/500 ⇒ ≈974 | yes | no | confirmed |
| 13 | seed robustness `wad` 862/855/875/868 | yes | no | confirmed |
| 14 | seed robustness `wad_adapt` 927/919/930/928 | yes | no | confirmed |
| 15 | no `d_i<C_i` | yes | no | confirmed |
| 16 | overspend Σd/D≈2.2 | exactly 2.200 | no | confirmed |
| 17 | "`g=1,λ=1` = `pd_global`" (§2.3) | no (728≠744) | yes | **refuted** |
| 18 | position ramp "868–875 across seeds" | no (seed 7: 848) | yes | **refuted** |
| 19 | "budget-normalised λ=1 ≈762" | no (748/775) | yes | **refuted/uncertain** |
| 20 | back-loading cost gradient (§2.2) | not shown | equal cost | **uncertain** |
| 21 | per-task/`[0.4,4.0]` evidence for the rule | wrong family | — | **mis-attributed** |
| 22 | "not exploiting the model" | n/a — exploit admitted | — | honest/confirmed |
| 23 | `pd_global` labelled "budget-feasible" | no (mean Σd/D=2.03) | yes | **refuted** |

---

## 9. Bottom line

* The **delivered rule is real and its headline numbers are exact**: `wad=875`
  (one pass) and `wad_adapt=930` (seed + 3 bounded per-flow passes), with the
  full per-level table reproduced. The core analysis levers it rests on
  (§1.1, §1.2, §1.4, §2.1) are correct.
* The **derivation is not as tight as presented**: the shape/magnitude constants
  are located empirically (the doc says so), and several *supporting* claims are
  wrong (`g=1,λ=1=pd_global`; the "868–875 across seeds" range; the "762"
  budget-normalised figure; the back-loading cost gradient). None of these
  change the delivered rule or its measured score, but they should not be
  presented as derived or reproduced.
* `wad_adapt` sits at the edge of the brief's "non-iterative" ideal: it is a
  fixed function, but it runs **3 full response-time analyses per instance** and
  adapts per flow, and it **exceeds the AGENTS headroom reference (~898)**. This
  is permitted ("a few passes") but should be stated plainly rather than
  positioned as a pure closed form — the honest description is "closed-form seed
  + 3 fixed-point correction passes", closer to a small fixed-point iteration
  than to `pd`.
* Finally, the metric is **extremely sensitive to floating-point formulation**;
  the report's "exact" counts should be read as exact *for the shipped code*.

## Addendum — `wad_adapt` seed robustness

Direct run (`algorithms.wad_adapt` over seeds 1/7/42/123, 1000 instances each):

| seed | my run | DERIVATION §4 | match |
|---|---|---|---|
| 1 | **927** | 927 | ✅ |
| 7 | **919** | 919 | ✅ |
| 42 | **930** | 930 | ✅ |
| 123 | **928** | 928 | ✅ |

The §4 `wad_adapt` row reproduces **exactly** (927/919/930/928), both from the
registered module and from `tmp/wad_adapt2.py` (`wad_P3_a0.5`). `tmp/wad_adapt2.py`
also confirms §5's pass/α claims: seed 42 gives
`wad_P2=923, wad_P3=926, wad_P4=928, wad_P3_a0.5=930, wad_P3_a0.7=920`, i.e.
`P=2≈923`, `P=3≈926–930`, `P=4≈928`, and `α=0.5` (the registered value) is
better than `α=0.6/0.7`. **Confirmed.**
