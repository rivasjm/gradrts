"""Verifier: budget overspend, lambda family, budget-normalised shape."""

import copy
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness, pool  # noqa: E402


def wad_seed(system, Q=1.6, S=0.4):
    for flow in system.flows:
        n = len(flow.tasks)
        total = sum(t.wcet for t in flow.tasks) or 1.0
        for i, t in enumerate(flow.tasks):
            pos = i / (n - 1) if n > 1 else 1.0
            t.deadline = flow.deadline * (Q * t.wcet / total + S * pos)


def wad_seed_norm(system, Q=1.6, S=0.4):
    """Same shape, but normalise each flow so sum_i d_i == D_f."""
    for flow in system.flows:
        n = len(flow.tasks)
        total = sum(t.wcet for t in flow.tasks) or 1.0
        raw = [Q * t.wcet / total + (S * i / (n - 1) if n > 1 else 1.0)
               for i, t in enumerate(flow.tasks)]
        z = sum(raw)
        for i, t in enumerate(flow.tasks):
            t.deadline = flow.deadline * raw[i] / z


def lambda_pd(lam):
    def rule(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            for t in flow.tasks:
                t.deadline = lam * t.wcet * flow.deadline / total
    return rule


def overspend():
    systems = pool.get_systems()
    ratios = []
    for u in pool.UTILIZATIONS:
        for base in systems:
            work = copy.deepcopy(base)
            pool.set_utilization(work, u)
            wad_seed(work)
            for f in work.flows:
                ratios.append(sum(t.deadline for t in f.tasks) / f.deadline)
    ratios.sort()
    print(f"wad seed flow-budget ratio sum(d)/D: "
          f"min={ratios[0]:.3f} median={ratios[len(ratios)//2]:.3f} "
          f"max={ratios[-1]:.3f} mean={sum(ratios)/len(ratios):.3f}")


def main():
    print("== budget overspend (wad seed) ==")
    overspend()
    print("\n== lambda family (scale of pd) ==")
    lambdas = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    rep = harness.evaluate(techniques={f"l{v}": lambda_pd(v) for v in lambdas})
    print("  " + "  ".join(f"l{v}={rep['techniques'][f'l{v}']['schedulable']}"
                           for v in lambdas))
    print("\n== wad seed vs budget-normalised seed ==")
    rep = harness.evaluate(techniques={"wad": wad_seed, "wad_norm": wad_seed_norm})
    print(f"  wad={rep['techniques']['wad']['schedulable']}  "
          f"wad_norm={rep['techniques']['wad_norm']['schedulable']}")


if __name__ == "__main__":
    main()
