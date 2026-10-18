"""Control for the shape ablation: same per-flow spend, ramp vs flat.

Reads a deadline dump, keeps each flow's total spend ``sum_i d_i`` and
redistributes it proportionally to ``C_i`` (gamma = 0). Tests the flat
assignment with the exact analysis. Analysis only: no search, no assignment.

Usage: .venv/bin/python workspace/edf_wad/flatten_check.py <dump-dir> ...
"""
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from workspace.framework_paper.systems import get_systems
from examples.generator import to_edf, set_utilization

HERE = os.path.dirname(os.path.abspath(__file__))
UTIL = np.linspace(0.5, 0.9, 20)
THREADS = 6


def load(path):
    per = {}
    with open(os.path.join(path, "deadlines.csv")) as f:
        for r in csv.DictReader(f):
            key = (r["u_index"], r["sys"])
            per.setdefault(key, []).append(
                (int(r["flow"]), int(r["pos"]), float(r["C"]), float(r["d"])))
    return per


def step(job):
    u_index, u, system, rows = job
    total = {}
    wcet = {}
    for flow, pos, c, d in rows:
        total[flow] = total.get(flow, 0.0) + d
        wcet[flow] = wcet.get(flow, 0.0) + c
    for flow, pos, c, d in rows:
        system.flows[flow].tasks[pos].deadline = total[flow] * c / wcet[flow]
    t0 = time.perf_counter()
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return u_index, int(system.is_schedulable()), time.perf_counter() - t0


def run(path):
    per = load(path)
    size = int(os.path.basename(path.rstrip("/")).rsplit("-", 1)[1])
    systems = get_systems(size)
    for s in systems:
        to_edf(s)
    counts = np.zeros(len(UTIL), dtype=int)
    times = np.zeros(len(UTIL))
    for u_index, u in enumerate(UTIL):
        for s in systems:
            set_utilization(s, u)
        jobs = [(u_index, u, s, per[(str(u_index), s.name)]) for s in systems]
        with Pool(THREADS) as pool:
            for ui, ok, el in pool.imap_unordered(step, jobs):
                counts[ui] += ok
                times[ui] += el
        print(f"{os.path.basename(path)} flat u={u:.3f} sched={counts[u_index]}",
              flush=True)
    print(f"{os.path.basename(path)} FLAT total {counts.sum()} / 1000, "
          f"mean analysis {times.mean():.4f}s", flush=True)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        run(p if os.path.isabs(p) else os.path.join(HERE, p))
