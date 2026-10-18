"""Q2: decompose order vs magnitude.

- magnitude at fixed order: lambda scaling of pd.
- order at fixed multiset: permute each flow's pd shares.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa


def pd_perm(perm):
    """Assign flow shares in the given permutation order (sum preserved)."""
    def f(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            ds = [t.wcet * flow.deadline / total for t in flow.tasks]
            for t, k in zip(flow.tasks, perm):
                t.deadline = ds[k]
    f.__name__ = f"perm{perm}"
    return f


PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

print("--- order effect: all 6 permutations of the pd share multiset ---")
run({f"p{p}": pd_perm(p) for p in PERMS})

print("\n--- magnitude effect: lambda (order fixed) ---")
run({f"l{lam}": pd_lambda(lam) for lam in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25]})

print("\n--- per-level: pd, best-reverse order, lambda2, gamma0.5 ---")
run({"pd": pd, "rev": pd_perm((2, 1, 0)), "lambda2": pd_lambda(2.0),
     "gamma0.5": pd_gamma(0.5)}, per_level=True)
