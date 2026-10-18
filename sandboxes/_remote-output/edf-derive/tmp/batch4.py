"""Scratch: utilisation-adaptive scale q(u)=alpha+beta*u, cumulative weight s.

    ../../../.venv/bin/python tmp/batch4.py
"""

import itertools
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402


def make(alpha, beta, s):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            cum = 0.0
            for t in flow.tasks:
                cum += t.wcet
                u = t.processor.utilization
                q = alpha + beta * u
                t.deadline = flow.deadline * (q * t.wcet + s * cum) / total
    return rule


if __name__ == "__main__":
    Alphas = [-0.5, 0.0, 0.5, 1.0]
    Betas = [1.0, 2.0, 3.0, 4.0, 5.0]
    Ss = [0.3, 0.6]
    techs = {}
    for a, b, s in itertools.product(Alphas, Betas, Ss):
        techs[f"a{a}_b{b}_s{s}"] = make(a, b, s)
    report = harness.evaluate(techniques=techs, population=25)
    rows = sorted(report["techniques"].items(),
                  key=lambda kv: -kv[1]["schedulable"])
    for name, m in rows[:15]:
        print(f"  {name:20s} {m['schedulable']:5d} / {report['instances']}")
    # reference: util-independent q=1.4 s=0.3
    ref = harness.evaluate(techniques={"ref": make(0.0, 0.0, 0.3)}, population=25)
    # patch q=1.4 explicitly
    def ref_rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            cum = 0.0
            for t in flow.tasks:
                cum += t.wcet
                t.deadline = flow.deadline * (1.4 * t.wcet + 0.3 * cum) / total
    ref = harness.evaluate(techniques={"q1.4_s0.3": ref_rule}, population=25)
    print("  ref q1.4_s0.3       ", ref["techniques"]["q1.4_s0.3"]["schedulable"],
          "/", ref["instances"])
