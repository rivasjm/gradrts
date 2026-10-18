"""Consolidated probes for the WAD derivation (unique name, seed-safe).

    ../../../.venv/bin/python tmp/wad_probe.py

Prints:
  A. two independent single-task flows on one processor: (R_a, R_b) vs (d_a, d_b)
  B. the eq(1) demand cap: R_a vs the interferer's local deadline d_t
  C. chain cost propagation: end-to-end R_1 vs an interferer's deadline d_u
  D. instrumented detail of one analysed task
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import explain, format_detail  # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402
from frozen.model import (Flow, LinearSystem, Processor,  # noqa: E402
                          SchedulerType, Task)


def build(specs, procs=3):
    """specs: list of (flow_name, [(task_name, C, T, proc_index), ...])."""
    system = LinearSystem()
    ps = [Processor(name=f"p{i}", sched=SchedulerType.EDF) for i in range(procs)]
    system.add_procs(*ps)
    for (fname, tasks) in specs:
        period = tasks[0][2]
        flow = Flow(name=fname, period=period, deadline=1e6)
        for (tname, C, T, pi) in tasks:
            flow.add_tasks(Task(name=tname, wcet=C, processor=ps[pi]))
        system.add_flows(flow)
    return system


def run(system, limit=1e9):
    HolisticLocalEDFAnalysis(limit_factor=limit, reset=True).apply(system)
    return {t.name: t.wcrt for t in system.tasks}


def probe_A():
    print("A. two independent single-task flows on p0  (C_a=2, C_b=3, T=10)")
    ds = [2, 4, 6, 8, 10, 20]
    print("        d_b=" + "".join(f"{d:>9d}" for d in ds))
    for da in ds:
        row = []
        for db in ds:
            s = build([("fa", [("a", 2, 10, 0)]), ("fb", [("b", 3, 10, 0)])])
            s["fa"]["a"].deadline = da
            s["fb"]["b"].deadline = db
            r = run(s)
            row.append(f"({r['a']:.0f},{r['b']:.0f})")
        print(f"  d_a={da:>3d}: " + " ".join(f"{x:>9}" for x in row))


def probe_B():
    print("\nB. eq(1) demand cap: victim a (C=2,T=10,d_a=2) + interferer t")
    print("   (C=3,T=10); sweep the interferer's local deadline d_t")
    for dt in [0, 1, 2, 5, 9, 10, 11, 20, 50]:
        s = build([("fa", [("a", 2, 10, 0)]), ("ft", [("t", 3, 10, 0)])])
        s["fa"]["a"].deadline = 2
        s["ft"]["t"].deadline = dt
        r = run(s)
        print(f"   d_t={dt:>3d} -> R_a={r['a']:.0f}")


def probe_C():
    print("\nC. chain: flow t0->t1 (T=10, C=2,2) on p0->p1 + interferer u")
    print("   (C=3,T=10) on p1; sweep d_u (d0=2, d1=8). u is not a")
    print("   predecessor, so j_1 = R_0 is unchanged.")
    print("      d_u :   R_0    R_1 (end-to-end)")
    for du in [2, 5, 10, 20, 50]:
        s = build([("f", [("t0", 2, 10, 0), ("t1", 2, 10, 1)]),
                   ("fu", [("u", 3, 10, 1)])])
        s["f"]["t0"].deadline = 2
        s["f"]["t1"].deadline = 8
        s["fu"]["u"].deadline = du
        r = run(s)
        print(f"    {du:>3d}  :  {r['t0']:>4.0f}    {r['t1']:>4.0f}")


def probe_D():
    print("\nD. instrumented detail: victim a (C=2,T=10,d_a=2) + interferer")
    print("   t (C=4,T=10,d_t=50)")
    s = build([("fa", [("a", 2, 10, 0)]), ("ft", [("t", 4, 10, 0)])])
    s["fa"]["a"].deadline = 2
    s["ft"]["t"].deadline = 50
    _, det = explain(s, limit_factor=1e9)
    print(format_detail("a", det["a"]))
    print(format_detail("t", det["t"]))


if __name__ == "__main__":
    probe_A()
    probe_B()
    probe_C()
    probe_D()
