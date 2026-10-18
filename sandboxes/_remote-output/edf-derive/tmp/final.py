"""Scratch: final candidate comparison across seeds + remaining-work family.

    ../../../.venv/bin/python tmp/final.py
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402


def pd_rule(system):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        for t in flow.tasks:
            t.deadline = t.wcet * flow.deadline / total


def pd_global_rule(system):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        c = 0.0
        for t in flow.tasks:
            c += t.wcet
            t.deadline = c * flow.deadline / total


def lin(Q, S):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            prior = 0.0
            for t in flow.tasks:
                t.deadline = flow.deadline * (Q * t.wcet + S * prior) / total
                prior += t.wcet
    return rule


def rank(Q, S):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            n = len(flow.tasks)
            for i, t in enumerate(flow.tasks):
                t.deadline = flow.deadline * (Q * t.wcet / total + (S * i / (n - 1) if n > 1 else 0.0))
    return rule


def remain(K):
    """d_i = D_f - K * (remaining work after i)."""
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            cum = 0.0
            for t in flow.tasks:
                cum += t.wcet
                t.deadline = flow.deadline - K * (total - cum)
    return rule


if __name__ == "__main__":
    cand = {
        "pd": pd_rule, "pd_global": pd_global_rule,
        "lin_Q1.7_S0.3": lin(1.7, 0.3), "rank_Q1.4_S0.3": rank(1.4, 0.3),
        "rank_Q1.6_S0.4": rank(1.6, 0.4),
    }
    for K in [0.5, 1.0, 1.5, 2.0, 3.0]:
        cand[f"remain_K{K}"] = remain(K)
    for seed in [42, 1, 7, 123]:
        rep = harness.evaluate(techniques=cand, seed=seed)
        line = "  ".join(f"{n}={rep['techniques'][n]['schedulable']}" for n in cand)
        print(f"seed {seed:4d}: {line}")
