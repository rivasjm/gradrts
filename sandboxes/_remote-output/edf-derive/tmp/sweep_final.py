"""Consolidated candidate sweep (sandbox-edf-derive)."""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from frozen import harness  # noqa: E402


def share_scale(system, g, lam):
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


def mk(g, lam):
    return lambda system: share_scale(system, g, lam)


if __name__ == "__main__":
    cands = {f"g{g}_l{lam}": mk(g, lam)
             for g in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
             for lam in (1.75, 2.0, 2.25)}
    rep = harness.evaluate(cands)
    for s, n in sorted(((m["schedulable"], n) for n, m in rep["techniques"].items()), reverse=True):
        print(f"{n:12s} {s}")
    # per-level for a few
    for n in ("g0.0_l2.0", "g0.3_l2.0", "g0.5_l2.0", "g1.0_l2.0"):
        print(n, rep["techniques"][n]["per_level"])
