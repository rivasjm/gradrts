"""Scratch: test the idealised ordering model.

Idealised model (single activation, well-spaced deadlines):
for each processor, sort tasks by local deadline (tie-break by name); the
cumulative load S_a = sum of C over tasks at or before a in that order.
Then R_last(flow) = sum over the flow's tasks of S_i (the jitter telescopes).
Predicted schedulable iff R_last <= flow.deadline.

Compare predicted schedulability against the real analysis.

    ../../../.venv/bin/python tmp/ideal.py
"""

import copy
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


def ideal_schedulable(system):
    S = {}
    for proc in system.processors:
        ts = sorted(proc.tasks, key=lambda t: (t.deadline, t.name))
        cum = 0.0
        for t in ts:
            cum += t.wcet
            S[t.name] = cum
    for flow in system.flows:
        R = sum(S[t.name] for t in flow.tasks)
        if R > flow.deadline:
            return False
    return True


def make_rule(kind, param):
    def rule(system):
        P = pd_deadlines(system)
        G = {}
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            c = 0.0
            for t in flow.tasks:
                c += t.wcet * flow.deadline / total
                G[t.name] = c
        for t in system.tasks:
            if kind == "gamma":
                t.deadline = (1 - param) * P[t.name] + param * G[t.name]
            elif kind == "lambda":
                t.deadline = param * P[t.name]
            elif kind == "global":
                t.deadline = G[t.name]
            elif kind == "allsame":
                t.deadline = t.flow.deadline
    return rule


def compare(kind, param):
    systems = pool.get_systems()
    actual = 0
    ideal = 0
    both = 0
    for u in pool.UTILIZATIONS:
        for base in systems:
            work = copy.deepcopy(base)
            pool.set_utilization(work, u)
            s = copy.deepcopy(work)
            make_rule(kind, param)(s)
            a = harness.is_schedulable(s)
            i = ideal_schedulable(s)
            actual += a
            ideal += i
            both += (a and i)
        print(f"  {kind}={param} u={u:.3f} running actual={actual} ideal={ideal} both={both}",
              flush=True)


if __name__ == "__main__":
    for kind, param in [("gamma", 0.0), ("gamma", 0.5), ("gamma", 1.0),
                        ("lambda", 2.0), ("allsame", 0)]:
        actual = ideal = both = 0
        systems = pool.get_systems()
        for u in pool.UTILIZATIONS:
            for base in systems:
                work = copy.deepcopy(base)
                pool.set_utilization(work, u)
                s = copy.deepcopy(work)
                make_rule(kind, param)(s)
                a = harness.is_schedulable(s)
                i = ideal_schedulable(s)
                actual += a
                ideal += i
                both += (a and i)
        print(f"{kind}={param}: actual={actual} ideal={ideal} both={both}")
