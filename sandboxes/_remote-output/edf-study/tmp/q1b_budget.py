"""Q1b: refine budget-feasible shapes, verify sums, per-level detail."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

# --- sanity: verify each family respects/breaks the budget on one system ---
from frozen import pool
s = pool.get_systems(seed=42, population=1)[0]
pool.set_utilization(s, 0.75)
for name, f in [("pd", pd), ("gamma0.5", pd_gamma(0.5)),
                ("gamnorm0.3", pd_gamma_norm(0.3)),
                ("bl1.5", back_load(1.5)), ("pd_global", pd_global)]:
    f(s)
    ratios = []
    for flow in s.flows:
        tot = sum(t.deadline for t in flow.tasks)
        ratios.append(tot / flow.deadline)
    print(f"budget {name:12s}: sum(d_i)/D_flow per flow = "
          f"{[round(r,4) for r in ratios]}")

print("\n--- finer gamnorm grid ---")
run({f"gn{g:.2f}": pd_gamma_norm(g) for g in
     [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})

print("\n--- per-level: pd vs best budget vs overspending ---")
run({"pd": pd, "gn0.3": pd_gamma_norm(0.3), "gn0.5": pd_gamma_norm(0.5),
     "pd_global": pd_global, "gamma0.5": pd_gamma(0.5),
     "lambda2": pd_lambda(2.0)}, per_level=True)
