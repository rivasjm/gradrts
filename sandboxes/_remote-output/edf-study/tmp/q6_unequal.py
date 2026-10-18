"""Q6 (Q3 caveat): with UNEQUAL per-processor loads, does per-processor
adaptation of lambda beat a single global constant?

harness.evaluate cannot be used because it forces equal utilization.
Custom loop: each processor is scaled to its own target u.
"""

import sys, os
from copy import deepcopy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa
from frozen import pool
from frozen.analysis import HolisticLocalEDFAnalysis

N = 50
TARGETS = [0.60, 0.75, 0.90]   # proc0, proc1, proc2


def build_unequal(seed=42, population=N):
    systems = pool.get_systems(seed=seed, population=population)
    for s in systems:
        for p, u in zip(s.processors, TARGETS):
            pool.set_processor_utilization(p, u)
    return systems


def shape_scale_global(system, g, lam):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        ds, prev = [], 0.0
        for t in flow.tasks:
            sh = t.wcet * flow.deadline / total
            ds.append(sh + g * prev)
            prev += sh
        scale = lam * flow.deadline / sum(ds)
        for t, d in zip(flow.tasks, ds):
            t.deadline = d * scale


def shape_scale_perproc(system, g, a, b):
    """lambda per TASK from its own processor's utilization: lam = a + b*U_proc."""
    lam = {}
    for p in system.processors:
        lam[p.name] = max(0.0, a + b * p.utilization)
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        ds, prev = [], 0.0
        for t in flow.tasks:
            sh = t.wcet * flow.deadline / total
            ds.append(sh + g * prev)
            prev += sh
        for t, d in zip(flow.tasks, ds):
            s = lam[t.processor.name] * flow.deadline / sum(ds)
            t.deadline = d * s


def evaluate(techs, systems):
    res = {n: 0 for n in techs}
    for base in systems:
        for name, tech in techs.items():
            w = deepcopy(base)
            tech(w)
            HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(w)
            if w.is_schedulable():
                res[name] += 1
    return res


systems = build_unequal()
print(f"unequal targets {TARGETS}, {N} systems, total {N} instances")
techs = {
    "pd": lambda s: shape_scale_global(s, 0.0, 1.0),
    "global_g0.3_l2.0": lambda s: shape_scale_global(s, 0.3, 2.0),
    "global_g0.2_l2.0": lambda s: shape_scale_global(s, 0.2, 2.0),
    "global_g0.3_l2.5": lambda s: shape_scale_global(s, 0.3, 2.5),
    "perproc_g0.3_0+3u": lambda s: shape_scale_perproc(s, 0.3, 0.0, 3.0),
    "perproc_g0.3_1+2u": lambda s: shape_scale_perproc(s, 0.3, 1.0, 2.0),
    "perproc_g0.2_0+3u": lambda s: shape_scale_perproc(s, 0.2, 0.0, 3.0),
}
res = evaluate(techs, systems)
for n, v in res.items():
    print(f"  {n:22s} {v:4d}/{N}")

print("\n(also: equal-load sanity, all procs scaled to 0.75)")
sys_eq = pool.get_systems(seed=42, population=N)
for s in sys_eq:
    pool.set_utilization(s, 0.75)
res2 = evaluate({k: v for k, v in techs.items()}, sys_eq)
for n, v in res2.items():
    print(f"  {n:22s} {v:4d}/{N}")
