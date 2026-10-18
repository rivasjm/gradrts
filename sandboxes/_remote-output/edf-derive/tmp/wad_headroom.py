"""Scratch: own per-instance optimiser -> headroom reference (population=25).

Coordinate descent (multiplicative per-task perturbations) minimising total
flow tardiness, initialised at the wad_adapt deadlines.

    ../../../.venv/bin/python tmp/wad_headroom.py
"""

import copy
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness, pool  # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402

BIG = 1e9


def tardiness(system):
    HolisticLocalEDFAnalysis(limit_factor=BIG, reset=True).apply(system)
    return sum(max(0.0, f.wcrt - f.deadline) for f in system.flows
               if f.wcrt is not None)


def seed(system):
    Q, S = 1.6, 0.4
    for flow in system.flows:
        n = len(flow.tasks)
        total = sum(t.wcet for t in flow.tasks) or 1.0
        for i, t in enumerate(flow.tasks):
            pos = i / (n - 1) if n > 1 else 1.0
            t.deadline = flow.deadline * (Q * t.wcet / total + S * pos)
    for _ in range(3):
        HolisticLocalEDFAnalysis(limit_factor=BIG, reset=True).apply(system)
        for flow in system.flows:
            r = flow.wcrt
            if not r:
                continue
            phi = min(1.25, max(0.75, (flow.deadline / r) ** 0.5))
            for t in flow.tasks:
                t.deadline *= phi


def optimise(system, passes=2):
    seed(system)
    cur = tardiness(system)
    for _ in range(passes):
        improved = False
        for t in system.tasks:
            base = t.deadline
            for factor in (0.8, 0.9, 1.1, 1.25, 2.0):
                t.deadline = base * factor
                v = tardiness(system)
                if v < cur - 1e-9:
                    cur = v
                    base = t.deadline
                    improved = True
            t.deadline = base
        if not improved:
            break
    return cur


if __name__ == "__main__":
    systems = pool.get_systems(population=25)
    opt = wad = bl = 0
    n = 0
    for u in pool.UTILIZATIONS:
        for base in systems:
            work = copy.deepcopy(base)
            pool.set_utilization(work, u)
            n += 1
            s1 = copy.deepcopy(work); seed(s1)
            wad += harness.is_schedulable(s1)
            s2 = copy.deepcopy(work)
            opt += (optimise(s2) < 1e-6)
    print(f"instances={n}  wad_adapt={wad}  optimiser={opt}")
