"""Scratch: alternative shapes vs the best linear rule.

    ../../../.venv/bin/python tmp/batch5.py
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402


def lin(Q, S):
    """d_i = D_f (Q C_i + S cum_{i-1}) / sumC  (own-C coeff Q, prior-cum coeff S)."""
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            prior = 0.0
            for t in flow.tasks:
                t.deadline = flow.deadline * (Q * t.wcet + S * prior) / total
                prior += t.wcet
    return rule


def pglobal(l):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            cum = 0.0
            for t in flow.tasks:
                cum += t.wcet
                t.deadline = l * flow.deadline * cum / total
    return rule


def ranklin(Q, S):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            n = len(flow.tasks)
            for i, t in enumerate(flow.tasks):
                t.deadline = flow.deadline * (Q * t.wcet / total + S * i / (n - 1))
    return rule


def powprim(theta):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            z = sum((t.wcet / total) ** theta for t in flow.tasks)
            for t in flow.tasks:
                t.deadline = flow.deadline * (t.wcet / total) ** theta / z
    return rule


if __name__ == "__main__":
    techs = {
        "lin_Q1.7_S0.3": lin(1.7, 0.3),
        "lin_Q1.6_S0.2": lin(1.6, 0.2),
        "lin_Q1.8_S0.4": lin(1.8, 0.4),
        "lin_Q2.0_S0.5": lin(2.0, 0.5),
        "lin_Q1.5_S0.15": lin(1.5, 0.15),
        "lin_Q1.7_S0.0": lin(1.7, 0.0),
        "lin_Q0.0_S1.0": lin(0.0, 1.0),
        "lin_Q0.0_S0.5": lin(0.0, 0.5),
    }
    for l in [0.8, 1.0, 1.2, 1.5, 2.0]:
        techs[f"pglobal_l{l}"] = pglobal(l)
    for Q, S in [(1.4, 0.0), (1.4, 0.3), (1.4, 0.6), (2.0, 0.3)]:
        techs[f"rank_Q{Q}_S{S}"] = ranklin(Q, S)
    for th in [0.6, 0.8, 1.0, 1.3, 1.6]:
        techs[f"pow{th}"] = powprim(th)
    report = harness.evaluate(techniques=techs)
    rows = sorted(report["techniques"].items(),
                  key=lambda kv: -kv[1]["schedulable"])
    for name, m in rows:
        print(f"  {name:18s} {m['schedulable']:5d}")
