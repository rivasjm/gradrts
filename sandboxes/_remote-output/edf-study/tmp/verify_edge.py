"""Adversarial edge-case probes for the verification pass.

Checks:
 A. pd vs budget-normalised g=0 (gn0.0) deadline vectors: how big is the diff?
 B. sum(d_i)/D_flow for the "closed best" (g=0.3, lam=2.0) and gamma0.5/pd_global.
 C. does an equal split d_i=D/n reproduce AGENTS.md's "equal = 314"?
 D. 5x3x3 gamma/lambda curve with the SAME custom loop as Q5 (to check the
    NOTES Q5 "5x3x3 balanced" row values 219/216).
 E. broader budget-compliance check of the budget search (100 instances).
 F. _wi as a function of d_i, compared to the closed-form pd.
"""

import os
import sys
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from frozen import pool, harness               # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis as A  # noqa: E402
from common import pd, pd_global, pd_gamma, pd_lambda, pd_gamma_norm  # noqa: E402


def shape_scale(system, g, lam):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        ds, prev = [], 0.0
        for t in flow.tasks:
            s = t.wcet * flow.deadline / total
            ds.append(s + g * prev)
            prev += s
        scale = lam * flow.deadline / sum(ds)
        for t, d in zip(flow.tasks, ds):
            t.deadline = d * scale


print("=" * 70)
print("A. pd vs gn0.0 (budget-normalised g=0) deadline vectors")
print("=" * 70)
s = pool.get_systems(seed=42, population=1)[0]
pool.set_utilization(s, 0.75)
s2 = deepcopy(s)
pd(s)
pd_gamma_norm(0.0)(s2)
maxrel = 0.0
for t, u in zip(s.tasks, s2.tasks):
    if t.deadline:
        maxrel = max(maxrel, abs(t.deadline - u.deadline) / abs(t.deadline))
print(f"  max relative deadline difference pd vs gn0.0: {maxrel:.3e}")
print(f"  scales (=D/sum(shares)) per flow:")
for flow in s.flows:
    tot = sum(t.wcet for t in flow.tasks)
    shares = [t.wcet * flow.deadline / tot for t in flow.tasks]
    print(f"    {flow.name}: D/sum(shares)-1 = {flow.deadline/sum(shares)-1:+.3e}")
# where do the counts diverge (pd=708 vs gn0.0=710)?
rep = harness.evaluate({"pd": pd, "gn0.0": pd_gamma_norm(0.0)}, per_level=True) \
    if False else harness.evaluate({"pd": pd, "gn0.0": pd_gamma_norm(0.0)})
print(f"  totals: pd={rep['techniques']['pd']['schedulable']} "
      f"gn0.0={rep['techniques']['gn0.0']['schedulable']}")

print()
print("=" * 70)
print("B. overspend ratios sum(d_i)/D_flow (system 0, u=0.75, all flows)")
print("=" * 70)
def ratios(f):
    w = deepcopy(s)
    f(w)
    out = []
    for flow in w.flows:
        out.append(sum(t.deadline for t in flow.tasks) / flow.deadline)
    return out

for name, f in [("pd", pd), ("gamma0.5", pd_gamma(0.5)),
                ("pd_global", pd_global), ("lambda2", pd_lambda(2.0)),
                ("closed best g.3 l2", lambda x: shape_scale(x, 0.3, 2.0)),
                ("budget closed gn0.6", pd_gamma_norm(0.6))]:
    r = ratios(f)
    print(f"  {name:22s} {[round(x,4) for x in r]}  min={min(r):.3f} max={max(r):.3f}")

print()
print("=" * 70)
print("C. equal split d_i = D_flow/n  (AGENTS.md calls this 'equal = 314')")
print("=" * 70)
def equal_split(system):
    for flow in system.flows:
        n = len(flow.tasks)
        for t in flow.tasks:
            t.deadline = flow.deadline / n

rep = harness.evaluate({"equal_split": equal_split})
print(f"  equal_split = {rep['techniques']['equal_split']['schedulable']}/1000")

print()
print("=" * 70)
print("D. 5x3x3 gamma/lambda curve (same custom loop as Q5b 'shuffle=False')")
print("=" * 70)
from q5_general import gen_pool, make, curve  # noqa: E402
levels = [round(0.5 + 0.4 * i / 9, 3) for i in range(10)]
systems = gen_pool(5, 3, 3, population=25)
gc, total, _ = curve({f"g{g}": make(g, 1, norm=False) for g in [0.0, 0.2, 0.3, 0.4, 0.6]},
                     systems, levels)
lc, _, _ = curve({f"l{lam}": make(0, lam) for lam in [1.0, 1.5, 2.0, 2.5, 3.0]},
                 systems, levels)
print(f"  gamma : {gc}")
print(f"  lambda: {lc}")
print("  NOTES Q5 row '5x3x3 balanced' claims: pd=179, best gamma=219 (g0.3), best lambda=216 (l1.5)")

print()
print("=" * 70)
print("E. budget-compliance of budget search, 100 instances")
print("=" * 70)
from search import search_technique  # noqa: E402
tech = search_technique(budget=True, passes=1, mults=(0.5, 1.5, 2.0))
systems50 = pool.get_systems(seed=42, population=50)
bad = 0
n_checked = 0
viol_flows = []
for u in [0.70, 0.75, 0.80, 0.85, 0.90]:
    for si in range(20):
        w = deepcopy(systems50[si])
        pool.set_utilization(w, u)
        tech(w)
        n_checked += 1
        for flow in w.flows:
            tot = sum(t.deadline for t in flow.tasks)
            if tot > flow.deadline + 1e-6:
                bad += 1
                viol_flows.append((u, si, flow.name, tot / flow.deadline))
            for t in flow.tasks:
                if t.deadline > flow.deadline + 1e-6:
                    bad += 1
print(f"  checked {n_checked} instances; violations={bad}")
for v in viol_flows[:10]:
    print("   ", v)

print()
print("=" * 70)
print("F. _wi vs the closed-form pd, fine sweep of d_i")
print("=" * 70)
sA = pool.get_systems(seed=42, population=1)[0]
pool.set_utilization(sA, 0.75)
pd(sA)
A(limit_factor=10, reset=True).apply(sA)
i = sA.processors[0].tasks[0]
print(f"  i={i.name} C={i.wcet:.4f} T={i.period:.4f} j={i.jitter:.4f}")
D = 200.0
prev = None
viol = 0
for d in [0, 10, 50, 100, 150, 175, 190, 195, 199, 200, 201, 250, 400]:
    i.deadline = float(d)
    wi = A._wi(i, 50.0, D)
    pl = __import__("math").ceil((50.0 + i.jitter) / i.period)
    pd_pts = 0 if D < i.deadline else \
        __import__("math").floor((i.jitter + D - i.deadline) / i.period) + 1
    m = min(pl, pd_pts)
    expect = m * i.wcet if m > 0 else 0
    mono = "" if prev is None or wi <= prev + 1e-12 else "  <-- NON-MONOTONE!"
    if mono:
        viol += 1
    prev = wi
    print(f"    d_i={d:6.1f}  _wi={wi:9.4f}  formula={expect:9.4f}{mono}")
print(f"  non-monotone steps found: {viol}")
