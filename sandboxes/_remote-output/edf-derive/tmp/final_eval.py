"""Final candidate confirmation + per-level. Scratch."""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from frozen import harness  # noqa: E402
from tmp.adaptive2 import make  # noqa: E402

if __name__ == "__main__":
    cands = {
        "p2_a0.5_cl0.7-1.5": make(alpha=0.5, lo=0.7, hi=1.5, passes=2),
        "p2_a0.6_cl0.7-1.5": make(alpha=0.6, lo=0.7, hi=1.5, passes=2),
        "p2_a0.5_cl0.75-1.25": make(alpha=0.5, lo=0.75, hi=1.25, passes=2),
        "p2_a0.6_cl0.75-1.25": make(alpha=0.6, lo=0.75, hi=1.25, passes=2),
        "p1_a0.5_cl0.7-1.5": make(alpha=0.5, lo=0.7, hi=1.5, passes=1),
    }
    rep = harness.evaluate(cands)
    for n, m in rep["techniques"].items():
        print(f"{n:24s} {m['schedulable']}")
    print("per-level p2_a0.5_cl0.7-1.5:",
          rep["techniques"]["p2_a0.5_cl0.7-1.5"]["per_level"])
    print("per-level p2_a0.6_cl0.7-1.5:",
          rep["techniques"]["p2_a0.6_cl0.7-1.5"]["per_level"])
