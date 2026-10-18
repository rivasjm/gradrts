"""Dump local-deadline assignments for a given method, for the shape analysis.

Usage (from ``code/``):
    .venv/bin/python workspace/edf_wad/dump_deadlines.py <method> <size>

``method`` is one of ``wad``, ``wad_adapt``, ``wad_budget``, ``rta_cd``.
Writes ``<method>-<size>/`` with ``schedulables.csv``, ``times.csv`` and
``deadlines.csv`` (one row per task: utilisation level, system, flow, position,
``C``, ``d``, proportional share ``s``, cumulative share, flow deadline/period).
"""
import os
import sys
import time
import csv
from multiprocessing import Pool

import numpy as np

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.wad_assignment import Q as SEED_Q, S as SEED_S, WADAssignment
from assignment.wad_adapt_assignment import WADAdaptAssignment
from assignment.rta_cd_assignment import RTACDAssignment
from examples.generator import to_edf, set_utilization
from workspace.framework_paper.systems import get_systems

HERE = os.path.dirname(os.path.abspath(__file__))
UTIL = np.linspace(0.5, 0.9, 20)
THREADS = 6
TEST = HolisticLocalEDFAnalysis(limit_factor=1, reset=True)


def method(name, system):
    if name == "wad":
        return WADAssignment(), "wad"
    if name == "wad_adapt":
        return WADAdaptAssignment(), "wad_adapt"
    if name == "wad_budget":
        return WADAdaptAssignment(budget=True), "wad_budget"
    if name == "wad_adapt_g0":
        # Same correction passes, ramp off (gamma = 0), same seed Q as WAD.
        return WADAdaptAssignment(s=0.0), "wad_adapt_g0"
    if name == "wad_adapt_g0_eq":
        # Ramp off, and the seed spends the same per flow as WAD
        # (sum_i d_i/D_f = Q + S*n/2, which is 2.2 for n=3 and 2.6 for n=5).
        n = len(system.flows[0].tasks)
        return WADAdaptAssignment(q=SEED_Q + SEED_S * n / 2, s=0.0), "wad_adapt_g0_eq"
    if name == "rta_cd":
        return RTACDAssignment(), "rta_cd"
    raise SystemExit(f"unknown method {name!r}")


def step(args):
    name, u, u_index, system = args
    assign, _ = method(name, system)
    before = time.perf_counter()
    assign.apply(system)
    elapsed = time.perf_counter() - before
    TEST.apply(system)
    ok = int(system.is_schedulable())
    rows = []
    for fi, flow in enumerate(system.flows):
        total_c = sum(t.wcet for t in flow.tasks)
        if total_c <= 0:
            total_c = 1.0
        cum = 0.0
        for i, task in enumerate(flow.tasks):
            s = task.wcet * flow.deadline / total_c
            rows.append((u_index, u, system.name, fi, i, len(flow.tasks),
                         task.wcet, task.deadline, s, cum,
                         flow.deadline, flow.period))
            cum += s
    extra = (getattr(assign, "sweeps", None), getattr(assign, "cap_hit", None),
             getattr(assign, "timeout_hit", None))
    return u_index, system.name, ok, elapsed, rows, extra


def run(name, size):
    out = os.path.join(HERE, f"{name}-{size}")
    os.makedirs(out, exist_ok=True)
    systems = get_systems(size)
    for s in systems:
        to_edf(s)
    sched = np.zeros((len(UTIL),), dtype=int)
    times = np.zeros((len(UTIL),))
    dl_path = os.path.join(out, "deadlines.csv")
    diag_path = os.path.join(out, "diagnostics.csv")
    with open(dl_path, "w", newline="") as dlf, open(diag_path, "w", newline="") as dgf:
        w = csv.writer(dlf)
        w.writerow(["u_index", "u", "sys", "flow", "pos", "n", "C", "d", "s",
                    "cum_s", "D", "T"])
        d = csv.writer(dgf)
        d.writerow(["u", "sys", "sched", "seconds", "sweeps", "cap_hit", "timeout_hit"])
        for u_index, u in enumerate(UTIL):
            for s in systems:
                set_utilization(s, u)
            jobs = [(name, u, u_index, s) for s in systems]
            with Pool(THREADS) as pool:
                for ui, sysname, ok, el, rows, extra in pool.imap_unordered(step, jobs):
                    sched[ui] += ok
                    times[ui] += el
                    w.writerows(rows)
                    d.writerow([u, sysname, ok, el] + ["" if v is None else int(v)
                                                       for v in extra])
            dlf.flush()
            dgf.flush()
            print(f"{name}-{size} u={u:.3f} ({u_index + 1}/{len(UTIL)}) "
                  f"sched={sched[u_index]}", flush=True)
    times /= len(systems)
    with open(os.path.join(out, "schedulables.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", name])
        for u, c in zip(UTIL, sched):
            w.writerow([u, int(c)])
    with open(os.path.join(out, "times.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", name])
        for u, t in zip(UTIL, times):
            w.writerow([u, t])
    print(f"{name}-{size}: total {sched.sum()} / {len(UTIL) * len(systems)}, "
          f"mean time {times.mean():.4f}s", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    run(sys.argv[1], int(sys.argv[2]))
