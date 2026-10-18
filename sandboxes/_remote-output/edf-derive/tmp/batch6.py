"""Scratch: two-pass rules that use the analysis' own responses.

    ../../../.venv/bin/python tmp/batch6.py
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402


def set_lin(system, Q=1.7, S=0.3):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        prior = 0.0
        for t in flow.tasks:
            t.deadline = flow.deadline * (Q * t.wcet + S * prior) / total
            prior += t.wcet


def measure(system):
    HolisticLocalEDFAnalysis(limit_factor=1e9, reset=True).apply(system)
    return {t.name: t.wcrt for t in system.tasks}


def pass2_scale(target, lo, hi):
    def rule(system):
        set_lin(system)
        R = measure(system)
        for flow in system.flows:
            r = R[flow.tasks[-1].name]
            if r and r > 0:
                rho = max(lo, min(hi, target * flow.deadline / r))
                for t in flow.tasks:
                    t.deadline *= rho
    return rule


def pass2_jitter(Q, S, k):
    """Second pass subtracts a fraction of the measured jitter from each deadline."""
    def rule(system):
        set_lin(system, Q, S)
        R = measure(system)
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            prior = 0.0
            for t in flow.tasks:
                j = R[t.predecessors[0].name] if t.predecessors else 0.0
                t.deadline = flow.deadline * (Q * t.wcet + S * prior) / total - k * j
                prior += t.wcet
    return rule


if __name__ == "__main__":
    techs = {}
    for target in [0.8, 0.9, 0.95, 1.0]:
        techs[f"scale_t{target}"] = pass2_scale(target, 0.5, 2.0)
    for k in [0.3, 0.6, 1.0]:
        techs[f"jit_k{k}"] = pass2_jitter(1.7, 0.3, k)
    # reference single pass
    def ref(system):
        set_lin(system)
    techs["single_lin"] = ref
    report = harness.evaluate(techniques=techs)
    rows = sorted(report["techniques"].items(),
                  key=lambda kv: -kv[1]["schedulable"])
    for name, m in rows:
        print(f"  {name:14s} {m['schedulable']:5d}  errs={len(m['errors'])}")
