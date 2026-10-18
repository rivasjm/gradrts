"""Lean refinement. Scratch."""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from frozen import harness  # noqa: E402
from tmp.adaptive2 import make  # noqa: E402

CANDS = {
    "p2_a0.6_c0.75-1.25": make(alpha=0.6, lo=0.75, hi=1.25, passes=2),
    "p2_a0.6_c0.8-1.2": make(alpha=0.6, lo=0.8, hi=1.2, passes=2),
    "p2_a0.75_c0.75-1.25": make(alpha=0.75, lo=0.75, hi=1.25, passes=2),
    "p2_a0.65_c0.75-1.25": make(alpha=0.65, lo=0.75, hi=1.25, passes=2),
    "p3_a0.6_c0.75-1.25": make(alpha=0.6, lo=0.75, hi=1.25, passes=3),
    "p2_a0.6_c0.7-1.3": make(alpha=0.6, lo=0.7, hi=1.3, passes=2),
    "p2_a0.55_c0.8-1.2": make(alpha=0.55, lo=0.8, hi=1.2, passes=2),
    "p4_a0.6_c0.8-1.2": make(alpha=0.6, lo=0.8, hi=1.2, passes=4),
}

if __name__ == "__main__":
    rep = harness.evaluate(CANDS)
    res = sorted(((m["schedulable"], n) for n, m in rep["techniques"].items()), reverse=True)
    for s, n in res:
        print(f"{n:24s} {s}")
    best = res[0][1]
    print("best", best, rep["techniques"][best]["per_level"])
    print("--- seeds for best ---")
    for seed in (1, 7, 13, 42):
        r = harness.evaluate({best: CANDS[best]}, seed=seed)
        print(f"  seed{seed}: {r['techniques'][best]['schedulable']}")
