"""Q4: deterministic coordinate-descent search over local deadlines.

Objective: total tardiness T(d)=sum_f max(0,wcrt_f - D_f); schedulable iff 0.
`budget=True` keeps sum(d_i) <= D_flow per flow (projects after each candidate).
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402

_BIG = 1e9
# Cap on the response time used while *searching*: aborting above
# CAP*D_flow keeps the (otherwise very expensive) analysis of wild
# candidate assignments cheap; the abort surfaces as a large flat penalty.
_CAP = 3.0


def tardiness(system):
    HolisticLocalEDFAnalysis(limit_factor=_CAP, reset=True).apply(system)
    t = 0.0
    for flow in system.flows:
        wcrt = flow.wcrt
        if wcrt is None:
            return 1e18
        if wcrt > flow.deadline:
            t += wcrt - flow.deadline
    return t


def current(system):
    return [t.deadline for t in system.tasks]


def set_deadlines(system, vec):
    for task, d in zip(system.tasks, vec):
        task.deadline = max(0.0, d)


def project_budget(system):
    for flow in system.flows:
        tot = sum(t.deadline for t in flow.tasks)
        if tot > flow.deadline and tot > 0:
            k = flow.deadline / tot
            for t in flow.tasks:
                t.deadline *= k


def shape_scale(system, g=0.3, lam=2.0, norm=True):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        ds, prev = [], 0.0
        for t in flow.tasks:
            s = t.wcet * flow.deadline / total
            ds.append(s + g * prev)
            prev += s
        scale = (lam * flow.deadline / sum(ds)) if norm else lam
        for t, d in zip(flow.tasks, ds):
            t.deadline = d * scale


def search_technique(budget=False, g0=0.3, lam0=2.0, passes=2,
                     mults=(0.5, 0.75, 1.25, 1.5, 2.0, 3.0)):
    def f(system):
        if budget:
            shape_scale(system, g=0.6, lam=1.0, norm=True)
            project_budget(system)
        else:
            shape_scale(system, g=g0, lam=lam0, norm=True)
        best = current(system)
        best_val = tardiness(system)
        if best_val == 0.0:
            return
        for _ in range(passes):
            improved = False
            for idx in range(len(best)):
                base = best[idx]
                for m in mults:
                    trial = list(best)
                    trial[idx] = base * m
                    set_deadlines(system, trial)
                    if budget:
                        project_budget(system)
                    v = tardiness(system)
                    if v < best_val - 1e-9:
                        best_val = v
                        best = current(system)
                        improved = True
                set_deadlines(system, best)
                if best_val == 0.0:
                    break
            if not improved or best_val == 0.0:
                break
        set_deadlines(system, best)
        if budget:
            project_budget(system)
    return f
