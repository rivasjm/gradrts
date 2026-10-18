"""Q2d: does permuting the BEST budget-feasible shape help? (robustness of Q2)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa


def base_vec(flow, kind):
    total = sum(t.wcet for t in flow.tasks)
    ds, prev = [], 0.0
    for t in flow.tasks:
        s = t.wcet * flow.deadline / total
        if kind == "gn06":
            ds.append(s + 0.6 * prev)
        elif kind == "cum":
            ds.append(s + prev)
        else:
            ds.append(s)
        prev += s
    if kind == "gn06":
        k = flow.deadline / sum(ds)
        ds = [d * k for d in ds]
    return ds


def perm_of(kind, perm):
    def f(system):
        for flow in system.flows:
            ds = base_vec(flow, kind)
            for t, p in zip(flow.tasks, perm):
                t.deadline = ds[p]
    f.__name__ = f"{kind}{perm}"
    return f


PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
print("--- permutations of the best budget-feasible shape gn0.6 (sum fixed) ---")
run({f"gn06_{p}": perm_of("gn06", p) for p in PERMS})
print("\n--- permutations of the cumulative (pd_global) shape ---")
run({f"cum_{p}": perm_of("cum", p) for p in PERMS})
