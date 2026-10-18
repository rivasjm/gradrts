"""Scratch: per-level comparison of candidate rules.

    ../../../.venv/bin/python tmp/perlevel.py
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness, pool  # noqa: E402


def pd_deadlines(system):
    out = {}
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        for t in flow.tasks:
            out[t.name] = t.wcet * flow.deadline / total
    return out


def pd_global_deadlines(system):
    out = {}
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        c = 0.0
        for t in flow.tasks:
            c += t.wcet * flow.deadline / total
            out[t.name] = c
    return out


def gamma(g):
    def rule(system):
        P = pd_deadlines(system)
        G = pd_global_deadlines(system)
        for t in system.tasks:
            t.deadline = (1 - g) * P[t.name] + g * G[t.name]
    return rule


def lam(l):
    def rule(system):
        P = pd_deadlines(system)
        for t in system.tasks:
            t.deadline = l * P[t.name]
    return rule


def linear(a, b):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            cum = 0.0
            for t in flow.tasks:
                cum += t.wcet
                t.deadline = flow.deadline * (a * cum + b * t.wcet) / total
    return rule


if __name__ == "__main__":
    techs = {"pd": gamma(0.0), "gamma0.5": gamma(0.5), "lambda2": lam(2.0),
             "lin_a0.3_b1.4": linear(0.3, 1.4), "lin_a0.3_b1.5": linear(0.3, 1.5),
             "lin_a0.4_b1.4": linear(0.4, 1.4)}
    report = harness.evaluate(techniques=techs)
    names = list(techs)
    print("  u      " + "".join(f"{n[:14]:>15s}" for n in names))
    for i, u in enumerate(pool.UTILIZATIONS):
        print(f"  {u:.3f}  " + "".join(
            f"{report['techniques'][n]['per_level'][i]:15d}" for n in names))
    print("  TOTAL  " + "".join(
        f"{report['techniques'][n]['schedulable']:15d}" for n in names))
