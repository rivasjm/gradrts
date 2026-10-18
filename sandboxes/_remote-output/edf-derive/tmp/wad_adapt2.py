"""Scratch: choose pass count / alpha for the wad-correction rule; seed check.

    ../../../.venv/bin/python tmp/wad_adapt2.py
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402

BIG = 1e9


def seed_wad(system):
    Q, S = 1.6, 0.4
    for flow in system.flows:
        n = len(flow.tasks)
        total = sum(t.wcet for t in flow.tasks) or 1.0
        for i, t in enumerate(flow.tasks):
            pos = i / (n - 1) if n > 1 else 1.0
            t.deadline = flow.deadline * (Q * t.wcet / total + S * pos)


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def make(passes, alpha, lo=0.75, hi=1.25):
    def rule(system):
        seed_wad(system)
        for _ in range(passes):
            HolisticLocalEDFAnalysis(limit_factor=BIG, reset=True).apply(system)
            for flow in system.flows:
                r = flow.wcrt
                if not r:
                    continue
                phi = clip((flow.deadline / r) ** alpha, lo, hi)
                for t in flow.tasks:
                    t.deadline *= phi
    return rule


if __name__ == "__main__":
    cand = {
        "wad_1pass": make(0, 0.6),
        "wad_P2": make(2, 0.6),
        "wad_P3": make(3, 0.6),
        "wad_P4": make(4, 0.6),
        "wad_P3_a0.5": make(3, 0.5),
        "wad_P3_a0.7": make(3, 0.7),
    }
    totals = {n: 0 for n in cand}
    for seed in [42, 1, 7, 123]:
        rep = harness.evaluate(techniques=cand, seed=seed)
        line = "  ".join(f"{n}={rep['techniques'][n]['schedulable']}" for n in cand)
        print(f"seed {seed:4d}: {line}")
        for n in cand:
            totals[n] += rep["techniques"][n]["schedulable"]
    print("\nsum over 4 seeds (max 4000):")
    for n, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {n:14s} {t}")
