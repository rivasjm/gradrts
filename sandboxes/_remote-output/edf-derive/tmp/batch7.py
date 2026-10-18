"""Scratch: refine rank/linear families and check a second seed.

    ../../../.venv/bin/python tmp/batch7.py
"""

import itertools
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402


def rank(Q, S):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            n = len(flow.tasks)
            for i, t in enumerate(flow.tasks):
                t.deadline = flow.deadline * (Q * t.wcet / total
                                              + S * i / (n - 1))
    return rule


def combo(Q, S, R):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            n = len(flow.tasks)
            prior = 0.0
            for i, t in enumerate(flow.tasks):
                t.deadline = flow.deadline * (
                    Q * t.wcet / total + S * prior / total + R * i / (n - 1))
                prior += t.wcet
    return rule


if __name__ == "__main__":
    techs = {}
    for Q, S in itertools.product([1.2, 1.3, 1.4, 1.5, 1.6], [0.1, 0.2, 0.3, 0.4, 0.5]):
        techs[f"rank_Q{Q}_S{S}"] = rank(Q, S)
    report = harness.evaluate(techniques=techs)
    rows = sorted(report["techniques"].items(),
                  key=lambda kv: -kv[1]["schedulable"])
    print("== rank family, seed 42 ==")
    for name, m in rows[:12]:
        print(f"  {name:16s} {m['schedulable']:5d}")

    # second seed check for a few candidates
    print("\n== seed generalisation ==")
    cand = {"rank_Q1.4_S0.3": rank(1.4, 0.3),
            "rank_Q1.3_S0.3": rank(1.3, 0.3),
            "rank_Q1.5_S0.3": rank(1.5, 0.3),
            "combo_Q1.4_S0.2_R0.15": combo(1.4, 0.2, 0.15),
            "combo_Q1.3_S0.2_R0.2": combo(1.3, 0.2, 0.2)}
    for seed in [42, 1, 7, 123]:
        rep = harness.evaluate(techniques=cand, seed=seed)
        line = "  ".join(f"{n}={rep['techniques'][n]['schedulable']}" for n in cand)
        print(f"  seed {seed:4d}: {line}")
