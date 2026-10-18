"""Scratch: finer grid around the best linear family.

    ../../../.venv/bin/python tmp/batch2.py
"""

import itertools
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402


def make_linear(a, b):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            cum = 0.0
            for t in flow.tasks:
                cum += t.wcet
                t.deadline = flow.deadline * (a * cum + b * t.wcet) / total
    return rule


if __name__ == "__main__":
    As = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    Bs = [1.2, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5]
    techs = {}
    for a, b in itertools.product(As, Bs):
        techs[f"a{a}_b{b}"] = make_linear(a, b)
    report = harness.evaluate(techniques=techs)
    rows = sorted(report["techniques"].items(),
                  key=lambda kv: -kv[1]["schedulable"])
    for name, m in rows[:25]:
        print(f"  {name:14s} {m['schedulable']:5d}")
