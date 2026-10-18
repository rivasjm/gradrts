"""Q4: closed form vs local search, schedulables vs cost.
population=10, 6 hard levels (u in [0.711,0.9]) = 60 instances.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from search import search_technique, shape_scale  # noqa

LEVELS = [0.711, 0.795, 0.879]
POP = 10


def closed_best(system):
    shape_scale(system, g=0.3, lam=2.0)


def budget_closed(system):
    shape_scale(system, g=0.6, lam=1.0)


techs = {
    "pd": pd,
    "lambda2": pd_lambda(2.0),
    "gamma0.5": pd_gamma(0.5),
    "closed_g.3_l2": closed_best,
    "budget_closed_gn.6": budget_closed,
    "search1": search_technique(budget=False, passes=1,
                                mults=(0.5, 1.5, 2.0)),
    "search2": search_technique(budget=False, passes=2,
                                mults=(0.5, 1.5, 2.0)),
    "search_budget2": search_technique(budget=True, passes=2,
                                       mults=(0.5, 1.5, 2.0)),
}

t0 = time.perf_counter()
report = run(techs, levels=LEVELS, population=POP, per_level=True)
print(f"\nwall time {time.perf_counter()-t0:.1f}s")
