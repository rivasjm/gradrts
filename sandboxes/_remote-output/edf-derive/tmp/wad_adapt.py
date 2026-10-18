"""Scratch: my own seed (wad) + bounded per-flow response-correction passes.

    ../../../.venv/bin/python tmp/wad_adapt.py
"""

import itertools
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


def seed_lin(system):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks) or 1.0
        prior = 0.0
        for t in flow.tasks:
            t.deadline = flow.deadline * (1.7 * t.wcet + 0.3 * prior) / total
            prior += t.wcet


def clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def make(seed, passes, alpha, lo, hi):
    def rule(system):
        seed(system)
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
    techs = {"wad_seed": seed_wad, "lin_seed": seed_lin}
    for alpha in [0.4, 0.6, 0.8]:
        for lo, hi in [(0.8, 1.25), (0.85, 1.15), (0.9, 1.1)]:
            techs[f"wad_a{alpha}_{lo}"] = make(seed_wad, 2, alpha, lo, hi)
    techs["wad_a0.6_P3"] = make(seed_wad, 3, 0.6, 0.8, 1.25)
    techs["lin_a0.6"] = make(seed_lin, 2, 0.6, 0.8, 1.25)
    report = harness.evaluate(techniques=techs)
    rows = sorted(report["techniques"].items(),
                  key=lambda kv: -kv[1]["schedulable"])
    for name, m in rows:
        print(f"  {name:20s} {m['schedulable']:5d}")
