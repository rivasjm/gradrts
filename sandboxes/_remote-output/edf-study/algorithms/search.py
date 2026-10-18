"""Budget-aware local search over local deadlines (Q4).

This module is intentionally **not registered**: it is orders of magnitude
slower than the closed forms (each call runs many full response-time analyses),
and registering it would make ``python run.py`` take many minutes. It backs the
Q4 finding and is driven from ``tmp/q4_compare.py``.

    from algorithms.search import make_search
    report = harness.evaluate({"search": make_search(budget=False)}, ...)

Objective (minimised): total tardiness
    T(d) = sum_flows max(0, flow.wcrt - flow.deadline)
using a huge ``limit_factor`` so WCRTs exist even for infeasible systems;
a system is schedulable iff T(d)=0. ``budget=True`` keeps sum(d_i) <= D_flow.
"""

from frozen.analysis import HolisticLocalEDFAnalysis

_BIG = 1e9
_CAP = 3.0  # abort above CAP*D_flow while searching (keeps analysis cheap)


def _tardiness(system):
    HolisticLocalEDFAnalysis(limit_factor=_CAP, reset=True).apply(system)
    total = 0.0
    for flow in system.flows:
        wcrt = flow.wcrt
        if wcrt is None:
            return 1e18
        if wcrt > flow.deadline:
            total += wcrt - flow.deadline
    return total


def _set(system, vec):
    for task, d in zip(system.tasks, vec):
        task.deadline = max(0.0, d)


def _current(system):
    return [t.deadline for t in system.tasks]


def _project(system):
    for flow in system.flows:
        tot = sum(t.deadline for t in flow.tasks)
        if tot > flow.deadline > 0:
            k = flow.deadline / tot
            for t in flow.tasks:
                t.deadline *= k


def _init(system, g=0.3, lam=2.0, norm=True):
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


def make_search(budget=False, g0=0.3, lam0=2.0, passes=2,
                mults=(0.5, 0.75, 1.25, 1.5, 2.0, 3.0)):
    """Return a technique doing greedy coordinate descent on deadlines."""
    def technique(system):
        if budget:
            _init(system, g=0.6, lam=1.0, norm=True)
            _project(system)
        else:
            _init(system, g0, lam0, norm=True)
        best = _current(system)
        best_val = _tardiness(system)
        if best_val == 0.0:
            return
        for _ in range(passes):
            improved = False
            for idx in range(len(best)):
                base = best[idx]
                for m in mults:
                    trial = list(best)
                    trial[idx] = base * m
                    _set(system, trial)
                    if budget:
                        _project(system)
                    val = _tardiness(system)
                    if val < best_val - 1e-9:
                        best_val = val
                        best = _current(system)
                        improved = True
                _set(system, best)
                if best_val == 0.0:
                    break
            if not improved or best_val == 0.0:
                break
        _set(system, best)
        if budget:
            _project(system)
    return technique
