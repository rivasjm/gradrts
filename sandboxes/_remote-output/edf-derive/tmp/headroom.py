"""Headroom reference: coordinate-descent over local deadlines (unregistered).

Also tests a two-pass slack-adaptive rule. Not imported by the harness.
"""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402
from frozen import harness  # noqa: E402

CAP = 4.0
MULTS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)


def tardiness(system):
    HolisticLocalEDFAnalysis(limit_factor=CAP, reset=True).apply(system)
    tot = 0.0
    for f in system.flows:
        r = f.wcrt
        if r is None:
            return 1e18
        if r > f.deadline:
            tot += r - f.deadline
    return tot


def shape(system, g=0.3, lam=2.0):
    for f in system.flows:
        total = sum(t.wcet for t in f.tasks)
        ds, prev = [], 0.0
        for t in f.tasks:
            s = t.wcet * f.deadline / total
            ds.append(s + g * prev)
            prev += s
        scale = lam * f.deadline / sum(ds)
        for t, d in zip(f.tasks, ds):
            t.deadline = d * scale


def make_search(passes=1):
    def tech(system):
        shape(system)
        best = [t.deadline for t in system.tasks]
        bv = tardiness(system)
        if bv == 0.0:
            return
        for _ in range(passes):
            improved = False
            for idx in range(len(best)):
                b = best[idx]
                for m in MULTS:
                    trial = list(best); trial[idx] = b * m
                    for t, d in zip(system.tasks, trial):
                        t.deadline = max(0.0, d)
                    v = tardiness(system)
                    if v < bv - 1e-9:
                        bv, b, improved = v, trial[idx], True
                best[idx] = b
                for t, d in zip(system.tasks, best):
                    t.deadline = max(0.0, d)
                if bv == 0.0:
                    break
            if not improved or bv == 0.0:
                break
        for t, d in zip(system.tasks, best):
            t.deadline = max(0.0, d)
    return tech


if __name__ == "__main__":
    rep = harness.evaluate({"headroom": make_search(passes=1)})
    print("headroom(full pool):", rep["techniques"]["headroom"]["schedulable"], "/1000")
    print("per-level:", rep["techniques"]["headroom"]["per_level"])
