"""Shared helpers for the deadline-assignment experiments.

Families are callables `f(system) -> None`; `run({name: f}, ...)` evaluates them
with `frozen.harness.evaluate` and prints totals + per-level counts.
"""

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from frozen import harness, pool  # noqa: E402

LEVELS = list(pool.UTILIZATIONS)


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #
def shares(flow):
    total = sum(t.wcet for t in flow.tasks)
    return [t.wcet * flow.deadline / total for t in flow.tasks], total


def pd(system):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        for t in flow.tasks:
            t.deadline = t.wcet * flow.deadline / total


def pd_global(system):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        cum = 0.0
        for t in flow.tasks:
            cum += t.wcet * flow.deadline / total
            t.deadline = cum


def equal(system):
    for flow in system.flows:
        for t in flow.tasks:
            t.deadline = flow.deadline


def period_prop(system):
    for flow in system.flows:
        total = sum(t.period for t in flow.tasks)
        for t in flow.tasks:
            t.deadline = flow.deadline * t.period / total


def pd_lambda(lam):
    def f(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            for t in flow.tasks:
                t.deadline = lam * t.wcet * flow.deadline / total
    f.__name__ = f"lam{lam}"
    return f


def pd_gamma(g):
    """d_i = share_i + g * (sum of previous shares)  (g=0 -> pd, g=1 -> pd_global)."""
    def f(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            prev = 0.0
            for t in flow.tasks:
                s = t.wcet * flow.deadline / total
                t.deadline = s + g * prev
                prev += s
    f.__name__ = f"gam{g}"
    return f


def pd_gamma_norm(g):
    """Same shape as pd_gamma but rescaled so sum(d_i) == D_flow (budget-feasible)."""
    def f(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            ds, prev = [], 0.0
            for t in flow.tasks:
                s = t.wcet * flow.deadline / total
                ds.append(s + g * prev)
                prev += s
            scale = flow.deadline / sum(ds)
            for t, d in zip(flow.tasks, ds):
                t.deadline = d * scale
    f.__name__ = f"gamnorm{g}"
    return f


def pd_reverse(system):
    """Same multiset of deadlines as pd, assigned in reverse order (order-only change)."""
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        ds = [t.wcet * flow.deadline / total for t in flow.tasks]
        for t, d in zip(flow.tasks, reversed(ds)):
            t.deadline = d


def back_load(beta):
    """Budget-feasible: weights w_k ~ beta**k, d_k = D_f * w_k / sum(w), later
    tasks (k increasing) get more when beta>1. Sum(d_i) == D_flow exactly."""
    def f(system):
        for flow in system.flows:
            n = len(flow.tasks)
            ws = [beta ** k for k in range(n)]
            sw = sum(ws)
            for t, w in zip(flow.tasks, ws):
                t.deadline = flow.deadline * w / sw
    f.__name__ = f"back{beta}"
    return f


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(techniques, levels=None, seed=pool.SEED, population=pool.POPULATION,
        size=15, per_level=False):
    levels = list(levels) if levels is not None else LEVELS
    report = harness.evaluate(techniques, seed=seed, size=size,
                              population=population, utilizations=levels)
    instances = report["instances"]
    print(f"=== seed={seed} size={size} systems={population} "
          f"levels={len(levels)} instances={instances} ===")
    print(f"  {'technique':18s} {'sched':>6s} {'ratio':>7s} {'tot_t':>8s}")
    for name, m in report["techniques"].items():
        print(f"  {name:18s} {m['schedulable']:6d} "
              f"{m['schedulable']/instances:7.1%} {m['total_time']:8.3f}")
    if per_level:
        print("  per level:")
        names = list(report["techniques"])
        print(f"  {'u':>7s}" + "".join(f"{n[:10]:>12s}" for n in names))
        for i, u in enumerate(levels):
            print(f"  {u:7.3f}" + "".join(
                f"{report['techniques'][n]['per_level'][i]:12d}" for n in names))
    return report


def timed(label, fn, *a, **k):
    t0 = time.perf_counter()
    out = fn(*a, **k)
    print(f"[{label}] {time.perf_counter()-t0:.2f}s", flush=True)
    return out
