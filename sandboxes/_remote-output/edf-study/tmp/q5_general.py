"""Q5: generalisation. Build pools with other sizes / mappings / deadline factors
(directly via pool.generate_system, since SIZES only holds 15) and see whether
the interior optimum survives.

Uses its own evaluation loop because harness.evaluate only knows SIZES.
"""

import os
import sys
from copy import deepcopy
from random import Random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from frozen import pool  # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402
from frozen.model import SchedulerType  # noqa: E402


def gen_pool(n_flows, n_tasks, n_procs, population=25, seed=42,
             dmin=0.5, dmax=1.0, pmin=100, pmax=1000, shuffle=False):
    rnd = Random(seed)
    systems = []
    for i in range(population):
        s = pool.generate_system(
            rnd, n_flows=n_flows, n_tasks=n_tasks, n_procs=n_procs,
            utilization=0.5, sched=SchedulerType.EDF,
            period_min=pmin, period_max=pmax,
            deadline_factor_min=dmin, deadline_factor_max=dmax)
        if shuffle:  # random mapping instead of the balanced round robin
            procs = s.processors
            for t in s.tasks:
                t.processor = rnd.choice(procs)
        systems.append(s)
    return systems


def share_gamma(flow, g):
    total = sum(t.wcet for t in flow.tasks)
    ds, prev = [], 0.0
    for t in flow.tasks:
        sh = t.wcet * flow.deadline / total
        ds.append(sh + g * prev)
        prev += sh
    return ds


def make(g=0.0, lam=1.0, norm=True):
    def f(system):
        for flow in system.flows:
            ds = share_gamma(flow, g)
            scale = lam * flow.deadline / sum(ds) if norm else lam
            for t, d in zip(flow.tasks, ds):
                t.deadline = d * scale
    return f


def evaluate(techniques, systems, utilizations):
    res = {name: [0] * len(utilizations) for name in techniques}
    for ui, u in enumerate(utilizations):
        for base in systems:
            work = deepcopy(base)
            pool.set_utilization(work, u)
            for name, tech in techniques.items():
                w = deepcopy(work)
                tech(w)
                HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(w)
                if w.is_schedulable():
                    res[name][ui] += 1
    return res


def report(tag, techniques, systems, utilizations):
    res = evaluate(techniques, systems, utilizations)
    total = len(systems) * len(utilizations)
    print(f"\n### {tag}  (systems={len(systems)} x levels={len(utilizations)}"
          f" = {total} instances)")
    print("  per-level (u : " + ", ".join(techniques.keys()) + ")")
    for ui, u in enumerate(utilizations):
        print(f"    u={u:.2f}: " + "  ".join(
            f"{n}={res[n][ui]}" for n in techniques))
    print("  totals: " + "  ".join(
        f"{n}={sum(res[n])}/{total}" for n in techniques))
    return res


def curve(techniques, systems, utilizations):
    res = evaluate(techniques, systems, utilizations)
    total = len(systems) * len(utilizations)
    return {n: sum(res[n]) for n in techniques}, total, res


if __name__ == "__main__":
    levels = [round(0.5 + 0.4 * i / 9, 3) for i in range(10)]

    print("*** same size 5x3x3, 25 systems, to calibrate the custom loop ***")
    pool15 = gen_pool(5, 3, 3, population=25)
    report("5x3x3 seed42 (custom loop)", {"pd": make(0, 1), "gamma.5": make(0.5, 1, norm=False)}, pool15, levels)

    for (nf, nt, np_) in [(4, 3, 2), (6, 3, 3), (4, 4, 3), (8, 2, 4), (3, 5, 3)]:
        systems = gen_pool(nf, nt, np_, population=25)
        gcurve, total, _ = curve({f"g{g}": make(g, 1, norm=False)
                                  for g in [0.0, 0.3, 0.5, 0.7]},
                                 systems, levels)
        lcurve, _, _ = curve({f"l{lam}": make(0.0, lam) for lam in [1.0, 1.5, 2.0, 2.5, 3.0]},
                             systems, levels)
        print(f"\n### size ({nf},{nt},{np_}) total={total}")
        print("  gamma:", {k: v for k, v in gcurve.items()})
        print("  lambda:", {k: v for k, v in lcurve.items()})
