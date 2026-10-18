"""Scratch: refine rank(Q,S) across seeds.

    ../../../.venv/bin/python tmp/refine.py
"""

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
                pos = (i / (n - 1)) if n > 1 else 1.0
                t.deadline = flow.deadline * (Q * t.wcet / total + S * pos)
    return rule


if __name__ == "__main__":
    pts = [(1.5, 0.35), (1.5, 0.4), (1.6, 0.4), (1.6, 0.35),
           (1.4, 0.4), (1.5, 0.45), (1.7, 0.4)]
    cand = {f"Q{q}_S{s}": rank(q, s) for (q, s) in pts}
    totals = {n: 0 for n in cand}
    for seed in [42, 1, 7, 123]:
        rep = harness.evaluate(techniques=cand, seed=seed)
        line = "  ".join(f"{n}={rep['techniques'][n]['schedulable']}" for n in cand)
        print(f"seed {seed:4d}: {line}")
        for n in cand:
            totals[n] += rep["techniques"][n]["schedulable"]
    print("\nsum over 4 seeds (max 4000):")
    for n, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  {n:12s} {t}")
