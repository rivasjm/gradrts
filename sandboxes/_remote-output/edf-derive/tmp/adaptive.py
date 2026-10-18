"""Two-pass, analysis-aware adaptive rules (scratch, unregistered)."""
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


def make_adaptive(alpha=0.5, lo=0.4, hi=4.0, passes=1, g=0.3, lam=2.0):
    def tech(system):
        shape(system, g, lam)
        for _ in range(passes):
            HolisticLocalEDFAnalysis(limit_factor=BIG, reset=True).apply(system)
            for f in system.flows:
                r = f.wcrt
                if not r:
                    continue
                phi = (f.deadline / r) ** alpha
                phi = min(hi, max(lo, phi))
                for t in f.tasks:
                    t.deadline *= phi
        # renormalise nothing
    return tech


if __name__ == "__main__":
    cands = {}
    for alpha in (0.25, 0.5, 0.75, 1.0):
        for lo, hi in ((0.4, 4.0), (0.7, 1.5)):
            cands[f"a{alpha}_{lo}_{hi}"] = make_adaptive(alpha=alpha, lo=lo, hi=hi, passes=1)
    for name, t in [("base", lambda s: shape(s)), ]:
        cands[name] = t
    rep = harness.evaluate(cands)
    for s, n in sorted(((m["schedulable"], n) for n, m in rep["techniques"].items()), reverse=True):
        print(f"{n:16s} {s}")
