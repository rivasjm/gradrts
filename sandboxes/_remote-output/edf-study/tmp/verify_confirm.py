"""Confirm the suspected NOTES errors and inspect the budget-search solution."""

import os
import sys
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from frozen import pool  # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis as A  # noqa: E402
from algorithms.shape import pd_cum_scaled, pd_budget_cum  # noqa: E402

print("=== overspend of the REGISTERED pd_cum_scaled (g=0.3, lam=2.0) ===")
for u in [0.6, 0.75, 0.9]:
    s = pool.get_systems(seed=42, population=1)[0]
    pool.set_utilization(s, u)
    pd_cum_scaled(s)
    r = [sum(t.deadline for t in f.tasks) / f.deadline for f in s.flows]
    print(f"  u={u}: sum(d_i)/D_flow per flow = {[round(x,4) for x in r]}")
print("  (NOTES Q1 table says 'closed best (g=0.3,lam=2)': x1.3-1.9;")
print("   shape.py docstring says '~1.3-1.6')")

print()
print("=== budget-search solution shape on a HARD instance ===")
from search import search_technique  # noqa: E402
sys50 = pool.get_systems(seed=42, population=50)
s = deepcopy(sys50[0])
pool.set_utilization(s, 0.9)
# pd shares for reference
tot = [sum(t.wcet for t in f.tasks) for f in s.flows]
shares = [[t.wcet * f.deadline / sum(tt.wcet for tt in f.tasks) for t in f.tasks]
          for f in s.flows]
tech = search_technique(budget=True, passes=1, mults=(0.5, 1.5, 2.0))
tech(s)
for fi, f in enumerate(s.flows):
    ds = [t.deadline for t in f.tasks]
    print(f"  {f.name}: D={f.deadline:9.2f} shares={[round(x,1) for x in shares[fi]]}"
          f" search={[round(x,1) for x in ds]} sum/D={sum(ds)/f.deadline:.4f}")
A(limit_factor=1, reset=True).apply(s)
print(f"  schedulable after budget search: {s.is_schedulable()}")

print()
print("=== is the interior optimum present for the (4,3,2)/(4,4,3) rows too? ===")
from q5_general import gen_pool, make, curve  # noqa: E402
levels = [round(0.5 + 0.4 * i / 9, 3) for i in range(10)]
for (nf, nt, np_) in [(4, 3, 2), (4, 4, 3)]:
    systems = gen_pool(nf, nt, np_, population=25)
    gc, total, _ = curve({f"g{g}": make(g, 1, norm=False) for g in [0.0, 0.3, 0.5]},
                         systems, levels)
    print(f"  ({nf},{nt},{np_}) gamma: {gc}")
