"""Micro-probe: confirm where local deadlines enter the analysis.

Run with the repo venv from the sandbox root:

    python tmp/probe_wi.py

 1. Raw eq(1) interference `_wi(i, t, D)` while only the interferer's local
    deadline `d_i` varies -> weakly decreasing, saturating at 0 once d_i > D.
 2. Vary a single task's local deadline and report the WCRTs of the *other*
    tasks (interferer channel).
 3. Confirm the `d_a` cancellation in eq(9) for `psi_ab` candidates:
    r = w - psi + d_a + j_a is invariant in d_a when psi = (p-1)T_a + d_a.
"""

import os
import sys
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from frozen import pool                      # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis as A  # noqa: E402


def build(u=0.75, seed=42):
    system = pool.get_systems(seed=seed, population=1)[0]
    pool.set_utilization(system, u)
    return system


def set_pd(system):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        for t in flow.tasks:
            t.deadline = t.wcet * flow.deadline / total


def analyzed(u=0.75):
    system = build(u)
    set_pd(system)
    A(limit_factor=10, reset=True).apply(system)
    return system


def main():
    system = analyzed()
    proc = system.processors[0]
    print("proc0 tasks (pd deadlines, u=0.75):")
    for t in proc.tasks:
        print(f"  {t.name:10s} C={t.wcet:8.4f} T={t.period:9.3f} "
              f"d={t.deadline:9.3f} jitter={t.jitter:.3f} wcrt={t.wcrt:.3f}")

    i = proc.tasks[0]
    print(f"\n--- eq(1) _wi(i={i.name}, t=50) vs interferer deadline d_i ---")
    print(f"i: C={i.wcet:.4f} T={i.period:.4f} jitter={i.jitter:.4f}")
    save = i.deadline
    for d in [0.0, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0]:
        i.deadline = d
        for D in [200.0, 1000.0]:
            wi = A._wi(i, 50.0, D)
            print(f"  d_i={d:9.1f}  D={D:7.1f}  _wi={wi:10.4f}")
    i.deadline = save

    # --- 2. interferer channel: vary one task's d_i, look at others' WCRT ---
    print(f"\n--- sweep d({i.name}); report all proc0 WCRTs ---")
    base = analyzed()
    proc = base.processors[0]
    tgt = proc.tasks[1] if len(proc.tasks) > 1 else proc.tasks[0]
    for mult in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        work = deepcopy(base)
        wi = [t for t in work.processors[0].tasks if t.name == tgt.name][0]
        wi.deadline = tgt.deadline * mult
        A(limit_factor=10, reset=True).apply(work)
        print(f"  d({tgt.name})={wi.deadline:9.3f}  "
              f"wcrt={[round(t.wcrt, 2) for t in work.processors[0].tasks]}")

    # --- 3. d_a cancellation check ---
    print("\n--- d_a cancellation for psi_ab candidates ---")
    system = analyzed()
    a = system.processors[0].tasks[0]
    for d_a in [5.0, 50.0, 500.0, 5000.0]:
        a.deadline = d_a
        p = 1
        psi = (p - 1) * a.period + a.deadline
        w = A._wab(a, psi, p, p * a.wcet)
        r = A._ra(a, psi, w)
        print(f"  d_a={d_a:8.1f}  psi={psi:9.2f}  w={w:9.3f}  r={r:9.3f}  "
              f"(w-(p-1)T+j = {w - (p-1)*a.period + a.jitter:.3f})")


if __name__ == "__main__":
    main()
