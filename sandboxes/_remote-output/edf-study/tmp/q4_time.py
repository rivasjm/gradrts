"""Time the search on a few hard instances to size the Q4 experiment."""

import sys, os, time
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from search import search_technique, shape_scale
from frozen import pool
from frozen.analysis import HolisticLocalEDFAnalysis

systems = pool.get_systems(seed=42, population=10)
for u in [0.795, 0.879, 0.9]:
    for si in range(3):
        s = deepcopy(systems[si])
        pool.set_utilization(s, u)
        shape_scale(s, 0.3, 2.0)
        HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(s)
        if s.is_schedulable():
            print(f"u={u} sys={si}: closed-form already schedulable (skip)")
            continue
        t0 = time.perf_counter()
        w = deepcopy(s)
        search_technique(budget=False, passes=1,
                         mults=(0.5, 1.5, 2.0))(w)
        dt = time.perf_counter() - t0
        HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(w)
        print(f"u={u} sys={si}: search passes=1 -> sched={w.is_schedulable()} "
              f"t={dt:.2f}s")
