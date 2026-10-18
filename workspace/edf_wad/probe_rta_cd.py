"""Probe: sweeps-to-convergence of RTA-CD, to choose a deterministic cap.

Design only. Does not touch the repo's class: subclasses it and replaces the
wall-clock stop by a sweep cap (plus a generous safety time cap).
"""
import os
import sys
import time
import csv
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from assignment.rta_cd_assignment import RTACDAssignment
from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from workspace.framework_paper.systems import get_systems
from examples.generator import to_edf, set_utilization

MAX_SWEEPS = 2000
MAX_SECONDS = 60.0


class ProbeRTACD(RTACDAssignment):
    def __init__(self, **kw):
        super().__init__(budget=1e9, **kw)
        self.sweeps = 0
        self.cap_hit = False
        self.converged = False

    def _descent(self, system, t0):
        best = self._violation(system)
        deadline = t0 + MAX_SECONDS
        self.sweeps = 0
        while True:
            if self.sweeps >= MAX_SWEEPS or time.perf_counter() > deadline:
                self.cap_hit = True
                return
            self.sweeps += 1
            improved = False
            for task in system.tasks:
                cur = task.deadline
                for fac in self.factors:
                    cand = cur * fac
                    if cand <= 1e-12:
                        continue
                    task.deadline = cand
                    o = self._violation(system)
                    if o < best - 1e-9:
                        best = o
                        cur = cand
                        improved = True
                task.deadline = cur
            if improved:
                if self._sched(system):
                    self.converged = True
                    return
            else:
                self.converged = True
                return


def run_one(args):
    size, u, idx = args
    system = get_systems(size)[idx]
    to_edf(system)
    set_utilization(system, u)
    a = ProbeRTACD()
    t0 = time.perf_counter()
    a.apply(system)
    return dict(size=size, u=u, sys=idx, seconds=time.perf_counter() - t0,
                sweeps=a.sweeps, cap=int(a.cap_hit), conv=int(a.converged),
                sched=int(system.is_schedulable()))


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe_rta_cd.csv")
    jobs = [(15, u, i) for u in (0.78, 0.86) for i in range(25)]
    jobs += [(25, u, i) for u in (0.78, 0.86) for i in range(25)]
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = list(ex.map(run_one, jobs))
    rows.sort(key=lambda r: -r["seconds"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows[:12]:
        print("size=%d u=%.2f sys=%2d t=%7.2fs sweeps=%4d cap=%d conv=%d sched=%d"
              % (r["size"], r["u"], r["sys"], r["seconds"], r["sweeps"],
                 r["cap"], r["conv"], r["sched"]))
    for size in (15, 25):
        rs = [r for r in rows if r["size"] == size]
        sw = np.array([r["sweeps"] for r in rs])
        tt = np.array([r["seconds"] for r in rs])
        print("size=%d n=%d sweeps median %.0f p90 %.0f p99 %.0f max %d | "
              "time median %.2f p90 %.2f max %.2f | capped %d | sched %d"
              % (size, len(rs), np.median(sw), np.percentile(sw, 90),
                 np.percentile(sw, 99), sw.max(), np.median(tt),
                 np.percentile(tt, 90), tt.max(),
                 sum(r["cap"] for r in rs), sum(r["sched"] for r in rs)))
    print("total wall %.1fs" % (time.perf_counter() - t0))
