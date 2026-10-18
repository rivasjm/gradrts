"""Scratch: per-instance search over local deadlines (headroom + structure probe).

Objective: sum over flows of the deadline violation max(0, R_last - D_f).
We minimise it. 0 => schedulable.

    ../../../.venv/bin/python tmp/search.py
"""

import copy
import math
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import pool  # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402


def violation(system):
    HolisticLocalEDFAnalysis(limit_factor=1e9, reset=True).apply(system)
    v = 0.0
    for flow in system.flows:
        r = flow.wcrt
        if r is None:
            return float("inf")
        v += max(0.0, r - flow.deadline)
    return v


def analyze(system):
    HolisticLocalEDFAnalysis(limit_factor=1e9, reset=True).apply(system)
    return [(flow.wcrt, flow.deadline) for flow in system.flows]


def pd_deadlines(system):
    out = {}
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        for t in flow.tasks:
            out[t.name] = t.wcet * flow.deadline / total
    return out


def set_deadlines(system, fractions):
    for flow in system.flows:
        for t in flow.tasks:
            t.deadline = fractions[t.name] * flow.deadline


def get_fractions(system):
    return {t.name: (t.deadline / t.flow.deadline) for t in system.tasks}


def search(system, passes=6, seed=0):
    import random
    rnd = random.Random(seed)

    # candidate inits (fractions of flow deadline)
    pd = pd_deadlines(system)
    P = {t.name: pd[t.name] / t.flow.deadline for t in system.tasks}
    cum = {}
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        c = 0.0
        for t in flow.tasks:
            c += t.wcet * flow.deadline / total
            cum[t.name] = c / flow.deadline
    G = cum
    inits = [dict(P), dict(G),
             {n: 0.5 * P[n] + 0.5 * G[n] for n in P},
             {n: 1.0 for n in P},
             {n: rnd.uniform(0.2, 1.5) for n in P}]

    best = None
    for init in inits:
        fr = dict(init)
        set_deadlines(system, fr)
        cur = violation(system)
        for _ in range(passes):
            improved = False
            names = list(fr)
            rnd.shuffle(names)
            for name in names:
                base = fr[name]
                for factor in (0.5, 0.7, 0.85, 1.18, 1.4, 2.0):
                    trial = dict(fr)
                    trial[name] = base * factor
                    set_deadlines(system, trial)
                    v = violation(system)
                    if v < cur - 1e-9:
                        cur = v
                        fr = trial
                        improved = True
                        break
            if not improved:
                break
        if best is None or cur < best[0]:
            best = (cur, dict(fr))
    return best


def run_structure(n_systems=8, levels=(0.5, 0.6, 0.7, 0.8, 0.9)):
    systems = pool.get_systems(population=pool.POPULATION)
    for u in levels:
        print(f"\n===== utilization {u:.2f} =====")
        for si in range(n_systems):
            base = copy.deepcopy(systems[si])
            pool.set_utilization(base, u)
            v0 = violation(copy.deepcopy(base))
            set_deadlines(base, {t.name: pd_deadlines(base)[t.name] / t.flow.deadline
                                 for t in base.tasks})
            vpd = violation(copy.deepcopy(base))
            work = copy.deepcopy(base)
            vbest, fr = search(work, seed=si)
            tag = "OPT" if vbest < 1e-6 else f"v={vbest:.2f}"
            print(f" sys{si}: pd_viol={vpd:8.3f} -> {tag}")
            if vbest < 1e-6 and vpd > 1e-6:
                set_deadlines(work, fr)
                print("     flow wcrt/deadline:", " ".join(
                    f"{r:.2f}/{d:.2f}" for r, d in analyze(work)))
                print("     fractions by flow:",
                      [[f"{fr[t.name]:.3f}" for t in flow.tasks] for flow in work.flows])


if __name__ == "__main__":
    run_structure(n_systems=int(sys.argv[1]) if len(sys.argv) > 1 else 8)
