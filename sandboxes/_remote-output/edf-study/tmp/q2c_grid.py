"""Q2c: 2-D closed-form grid over (shape g, magnitude lambda)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa


def shape_scale(g, lam):
    def f(system):
        for flow in system.flows:
            total = sum(t.wcet for t in flow.tasks)
            ds, prev = [], 0.0
            for t in flow.tasks:
                s = t.wcet * flow.deadline / total
                ds.append(s + g * prev)
                prev += s
            scale = lam * flow.deadline / sum(ds)
            for t, d in zip(flow.tasks, ds):
                t.deadline = d * scale
    f.__name__ = f"g{g}_l{lam}"
    return f


gs = [round(0.1 * i, 1) for i in range(0, 11)]
lams = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0]
techs = {}
for g in gs:
    for lam in lams:
        techs[f"g{g}_l{lam}"] = shape_scale(g, lam)

report = run(techs)

print("\n--- grid (schedulable), rows=g, cols=lambda ---")
print("        " + "".join(f"{lam:>7.2f}" for lam in lams))
for g in gs:
    print(f"g={g:4.1f} " + "".join(
        f"{report['techniques'][f'g{g}_l{lam}']['schedulable']:7d}" for lam in lams))

best = max(report["techniques"].items(), key=lambda kv: kv[1]["schedulable"])
print(f"\nbest: {best[0]} -> {best[1]['schedulable']}")
