"""Verify that search_budget respects sum(d_i) <= D_flow after the search."""

import sys, os
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from search import search_technique
from frozen import pool

systems = pool.get_systems(seed=42, population=50)
tech = search_technique(budget=True, passes=1, mults=(0.5, 1.5, 2.0))
bad = 0
for u in [0.75, 0.85, 0.9]:
    for si in range(5):
        s = deepcopy(systems[si])
        pool.set_utilization(s, u)
        tech(s)
        for flow in s.flows:
            tot = sum(t.deadline for t in flow.tasks)
            if tot > flow.deadline + 1e-6:
                bad += 1
                print(f"VIOLATION u={u} sys={si} flow={flow.name} "
                      f"sum/D={tot/flow.deadline:.6f}")
            for t in flow.tasks:
                if t.deadline > flow.deadline + 1e-6:
                    bad += 1
                    print(f"INDIVIDUAL VIOLATION u={u} sys={si} "
                          f"{t.name} d/D={t.deadline/flow.deadline:.6f}")
print(f"checked {15} instances x 5 flows; budget violations = {bad}")
