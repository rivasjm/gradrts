"""Scratch: reproduce the gamma/lambda families and test quick candidate rules.

    ../../../.venv/bin/python tmp/eval_families.py
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness  # noqa: E402


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
        cum = 0.0
        for t in flow.tasks:
            cum += t.wcet * flow.deadline / total
            out[t.name] = cum
    return out


def make_gamma_interp(gamma):
    def rule(system):
        P = pd_deadlines(system)
        G = pd_global_deadlines(system)
        for t in system.tasks:
            t.deadline = (1 - gamma) * P[t.name] + gamma * G[t.name]
    return rule


def make_gamma_lerp_to_D(gamma):
    """d = pd + gamma*(flow.deadline - pd): gamma=1 -> all = flow deadline."""
    def rule(system):
        P = pd_deadlines(system)
        for flow in system.flows:
            for t in flow.tasks:
                t.deadline = P[t.name] + gamma * (flow.deadline - P[t.name])
    return rule


def make_lambda(lam):
    def rule(system):
        P = pd_deadlines(system)
        for t in system.tasks:
            t.deadline = lam * P[t.name]
    return rule


def make_equal(_=None):
    def rule(system):
        for flow in system.flows:
            for t in flow.tasks:
                t.deadline = flow.deadline / len(flow.tasks)
    return rule


def make_all_D(_=None):
    def rule(system):
        for flow in system.flows:
            for t in flow.tasks:
                t.deadline = flow.deadline
    return rule


def run(techs, label):
    report = harness.evaluate(techniques=techs)
    print(f"--- {label}")
    for name, m in report["techniques"].items():
        print(f"    {name:22s} {m['schedulable']:5d}")


if __name__ == "__main__":
    run({f"g{i}": make_gamma_interp(i / 10) for i in range(11)}, "gamma interp pd->pd_global")
    run({f"gD{i}": make_gamma_lerp_to_D(i / 10) for i in range(11)},
        "gamma lerp pd->flow.D")
    run({f"l{v}": make_lambda(v) for v in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]},
        "lambda scale pd")
    run({"equal": make_equal(), "allD": make_all_D()}, "equal / allD")
