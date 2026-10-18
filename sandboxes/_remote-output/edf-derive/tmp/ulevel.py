"""Scratch: per-level grid over (q,s) to see if coefficients should adapt to u.

    ../../../.venv/bin/python tmp/ulevel.py
"""

import itertools
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402


def make(q, s):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            cum = 0.0
            for t in flow.tasks:
                cum += t.wcet
                t.deadline = flow.deadline * (q * t.wcet + s * cum) / total
    return rule


if __name__ == "__main__":
    Qs = [0.8, 1.2, 1.6, 2.0, 2.4, 3.0, 4.0]
    Ss = [0.0, 0.3, 0.6, 1.0, 1.5, 2.0]
    for u in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        techs = {f"q{q}_s{s}": make(q, s) for q, s in itertools.product(Qs, Ss)}
        report = harness.evaluate(techniques=techs, utilizations=[u])
        rows = sorted(report["techniques"].items(),
                      key=lambda kv: -kv[1]["schedulable"])
        top = "  ".join(f"({n}:{m['schedulable']})" for n, m in rows[:5])
        print(f"u={u:.2f} (max 50): {top}")
