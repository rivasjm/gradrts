"""Q2b: 2-D decomposition shape (fixed budget) x magnitude (scale).

pd_gamma_norm(g) fixes sum(d_i)=D_flow (a *shape*), and multiplying by lambda
adds a pure *magnitude* factor. pd = shape g=0, lambda=1. gamma0.5 is close to
shape g=0.5 at its natural magnitude.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa


def shape_scale(g, lam):
    """budget-normalized cumulative shape g, then scaled by lambda."""
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


for g in [0.0, 0.3, 0.5, 0.6, 1.0]:
    print(f"\n--- shape g={g}, magnitude lambda sweep ---")
    run({f"g{g}l{lam}": shape_scale(g, lam) for lam in
         [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]})
