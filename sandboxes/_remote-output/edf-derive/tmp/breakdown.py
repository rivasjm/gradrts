"""Scratch: full instrumented breakdown of one instance at high utilisation.

    ../../../.venv/bin/python tmp/breakdown.py [sys] [u]
"""

import copy
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import explain, format_detail, pool  # noqa: E402


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


def apply_gamma(system, gamma):
    P = pd_deadlines(system)
    G = pd_global_deadlines(system)
    for t in system.tasks:
        t.deadline = (1 - gamma) * P[t.name] + gamma * G[t.name]


def show(system, label):
    ok, details = explain(system, limit_factor=1000)
    print(f"===== {label}: schedulable={ok}")
    for flow in system.flows:
        print(f"  flow {flow.name}: R={flow.wcrt} D={flow.deadline:.2f}")
        for t in flow.tasks:
            print("    " + format_detail(t.name, details[t.name]).replace("\n", "\n    "))


if __name__ == "__main__":
    si = int(sys.argv[1]) if len(sys.argv) > 1 else 11
    u = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    base = pool.get_systems()[si]
    work = copy.deepcopy(base)
    pool.set_utilization(work, u)
    for g in (0.0, 0.5, 1.0):
        s = copy.deepcopy(work)
        apply_gamma(s, g)
        show(s, f"sys{si} u={u} gamma={g}")
        print()
