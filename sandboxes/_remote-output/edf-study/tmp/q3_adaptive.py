"""Q3: is the optimal (shape g, magnitude lambda) universal, or does it track u?

Steps:
 1. per-level grid over (g, lambda) -> optimal (g*, lambda*) per level.
 2. a smooth u-adaptive rule lambda(u) fitted from the trend, evaluated against
    the best global constant.
 3. a per-flow adaptive rule based on the flow's deadline slack.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from frozen.analysis import HolisticLocalEDFAnalysis

LEVELS = list(pool.UTILIZATIONS)


def shape_scale(g, lam):
    def f(system):
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
    f.__name__ = f"g{g}_l{lam}"
    return f


GS = [0.0, 0.2, 0.4, 0.6]
LAMS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]

print("=== per-level grid (g x lambda) ===")
report = run({f"{g}_{l}": shape_scale(g, l) for g in GS for l in LAMS},
             per_level=True)

print("\n=== best (g,lambda) per level ===")
lookup = {}
for i, u in enumerate(LEVELS):
    best = max(((g, l) for g in GS for l in LAMS),
               key=lambda gl: report["techniques"][f"{gl[0]}_{gl[1]}"]["per_level"][i])
    val = report["techniques"][f"{best[0]}_{best[1]}"]["per_level"][i]
    lookup[round(u, 3)] = best
    print(f"  u={u:.3f}: best g={best[0]} lam={best[1]} -> {val}/50")


# --- 2. smooth u-adaptive rule: lambda(u) = a + b*u, g fixed at best global ---
def adaptive(g, a, b):
    def f(system):
        u = system.utilization
        lam = max(0.5, a + b * u)
        shape_scale(g, lam)(system)
    f.__name__ = "adaptive"
    return f


print("\n=== adaptive lambda(u)=a+b*u vs global constants ===")
best_g, best_lam = max(
    ((g, l) for g in GS for l in LAMS),
    key=lambda gl: report["techniques"][f"{gl[0]}_{gl[1]}"]["schedulable"])
print(f"  (global best from grid: g={best_g} lam={best_lam} "
      f"-> {report['techniques'][f'{best_g}_{best_lam}']['schedulable']})")
run({"global_best": shape_scale(best_g, best_lam),
     "global_g0.5_l2.0": shape_scale(0.5, 2.0),
     "adapt_g0.4_0+2.5u": adaptive(0.4, 0.0, 2.5),
     "adapt_g0.4_0.5+2.2u": adaptive(0.4, 0.5, 2.2),
     "adapt_g0.3_0+2.8u": adaptive(0.3, 0.0, 2.8)})


# --- 3. per-flow adaptive lambda from the flow's own deadline slack ---
def flow_slack(D, n, T):
    return D / (n * T) if n * T > 0 else 0


def perflow(g, lam_lo, lam_hi):
    """lambda per flow interpolated by its deadline factor r = D/(n*T) in [.5,1]."""
    def f(system):
        for flow in system.flows:
            n = len(flow.tasks)
            r = flow_slack(flow.deadline, n, flow.period)
            lam = lam_lo + (lam_hi - lam_lo) * min(1.0, max(0.0, (r - 0.5) / 0.5))
            total = sum(t.wcet for t in flow.tasks)
            ds, prev = [], 0.0
            for t in flow.tasks:
                s = t.wcet * flow.deadline / total
                ds.append(s + g * prev)
                prev += s
            scale = lam * flow.deadline / sum(ds)
            for t, d in zip(flow.tasks, ds):
                t.deadline = d * scale
    f.__name__ = "perflow"
    return f


print("\n=== per-flow lambda from flow slack factor r=D/(n*T) ===")
run({"global_g0.4_l2.0": shape_scale(0.4, 2.0),
     "perflow_2.0_2.5": perflow(0.4, 2.0, 2.5),
     "perflow_1.5_2.5": perflow(0.4, 1.5, 2.5),
     "perflow_2.0_1.5": perflow(0.4, 2.0, 1.5),
     "perflow_1.5_3.0": perflow(0.4, 1.5, 3.0)})
