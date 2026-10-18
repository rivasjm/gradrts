"""Refine the two-pass adaptive rule; check seeds. Scratch, unregistered."""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402
from frozen import harness  # noqa: E402

BIG = 1e9


def shape(system, g=0.3, lam=2.0):
    for f in system.flows:
        total = sum(t.wcet for t in f.tasks)
        ds, prev = [], 0.0
        for t in f.tasks:
            s = t.wcet * f.deadline / total
            ds.append(s + g * prev)
            prev += s
        scale = lam * f.deadline / sum(ds)
        for t, d in zip(f.tasks, ds):
            t.deadline = d * scale


def make(alpha=0.6, lo=0.7, hi=1.5, passes=1, g=0.3, lam=2.0, per_task=False):
    def tech(system):
        shape(system, g, lam)
        for _ in range(passes):
            HolisticLocalEDFAnalysis(limit_factor=BIG, reset=True).apply(system)
            if per_task:
                for t in system.tasks:
                    r = t.wcrt
                    if r:
                        phi = min(hi, max(lo, (t.flow.deadline / r) ** alpha))
                        t.deadline *= phi
            else:
                for f in system.flows:
                    r = f.wcrt
                    if r:
                        phi = min(hi, max(lo, (f.deadline / r) ** alpha))
                        for t in f.tasks:
                            t.deadline *= phi
    return tech


if __name__ == "__main__":
    cands = {}
    for passes in (1, 2, 3):
        for alpha in (0.4, 0.6, 0.8):
            cands[f"p{passes}_a{alpha}"] = make(alpha=alpha, passes=passes)
    for lo, hi in ((0.6, 1.4), (0.75, 1.25), (0.8, 1.2), (0.7, 2.0), (0.5, 2.0)):
        cands[f"clamp{lo}_{hi}"] = make(lo=lo, hi=hi)
    cands["pertask"] = make(per_task=True)
    rep = harness.evaluate(cands)
    for s, n in sorted(((m["schedulable"], n) for n, m in rep["techniques"].items()), reverse=True):
        print(f"{n:16s} {s}")
    best = max(cands, key=lambda n: rep["techniques"][n]["schedulable"])
    print("best:", best, rep["techniques"][best]["per_level"])
    # seeds
    print("--- other seeds (p1_a0.6 default) ---")
    for seed in (1, 7, 13, 42):
        r = harness.evaluate({"r": make()}, seed=seed)
        print("seed", seed, r["techniques"]["r"]["schedulable"])
