"""Q4 full pool: budget-constrained vs unconstrained search over all 1000 instances."""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from search import search_technique, shape_scale  # noqa


def closed_best(system):
    shape_scale(system, g=0.3, lam=2.0)


techs = {
    "closed_g.3_l2": closed_best,
    "budget_closed_gn.6": lambda s: shape_scale(s, 0.6, 1.0),
    "search_unbounded": search_technique(budget=False, passes=1,
                                         mults=(0.5, 1.5, 2.0)),
    "search_budget": search_technique(budget=True, passes=1,
                                      mults=(0.5, 1.5, 2.0)),
}
t0 = time.perf_counter()
run(techs, per_level=True)
print(f"\nwall time {time.perf_counter()-t0:.1f}s")
