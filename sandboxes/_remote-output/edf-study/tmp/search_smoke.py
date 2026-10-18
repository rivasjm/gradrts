"""Smoke-test the search: time per system and whether it fixes infeasible ones."""

import sys, os, time
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from search import search_technique, tardiness
from frozen import pool
from frozen.analysis import HolisticLocalEDFAnalysis

# build a few hard instances near the boundary
systems = pool.get_systems(seed=42, population=50)
for u in [0.85, 0.9]:
    for si in [0, 1, 2]:
        s = deepcopy(systems[si])
        pool.set_utilization(s, u)
        pd(s)
        HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(s)
        base_sched = s.is_schedulable()
        t0 = time.perf_counter()
        for tech, name in [(search_technique(budget=True), "budget"),
                           (search_technique(budget=False), "unb")]:
            w = deepcopy(s)
            tech(w)
            HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(w)
            dt = time.perf_counter() - t0
            print(f"u={u} sys={si} pd_sched={base_sched} {name}: sched_after="
                  f"{w.is_schedulable()} tard={tardiness(w):.4f} t={dt:.2f}s")
            t0 = time.perf_counter()
