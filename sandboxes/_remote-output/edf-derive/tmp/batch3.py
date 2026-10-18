"""Scratch: 3-parameter search d_i = D_f*(p + q*C_i/sumC + s*cum_i/sumC).

    ../../../.venv/bin/python tmp/batch3.py
"""

import itertools
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402


def make(p, q, s):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            cum = 0.0
            for t in flow.tasks:
                cum += t.wcet
                t.deadline = flow.deadline * (p + q * t.wcet / total + s * cum / total)
    return rule


if __name__ == "__main__":
    Ps = [-0.2, 0.0, 0.2, 0.4]
    Qs = [1.2, 1.4, 1.6]
    Ss = [0.15, 0.3, 0.45]
    techs = {}
    for p, q, s in itertools.product(Ps, Qs, Ss):
        techs[f"p{p}_q{q}_s{s}"] = make(p, q, s)
    report = harness.evaluate(techniques=techs, population=20)
    rows = sorted(report["techniques"].items(),
                  key=lambda kv: -kv[1]["schedulable"])
    for name, m in rows[:15]:
        print(f"  {name:18s} {m['schedulable']:5d} / {report['instances']}")
