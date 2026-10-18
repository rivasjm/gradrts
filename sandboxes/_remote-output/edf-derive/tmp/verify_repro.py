"""Independent verifier probes. Run with the repo venv (Python 3.13)."""

import copy
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import harness, pool  # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402
from frozen.model import (Flow, LinearSystem, Processor,  # noqa: E402
                          SchedulerType, Task)


def build(specs, procs=3, deadline=1e9):
    system = LinearSystem()
    ps = [Processor(name=f"p{i}", sched=SchedulerType.EDF) for i in range(procs)]
    system.add_procs(*ps)
    for (fname, tasks) in specs:
        period = tasks[0][2]
        flow = Flow(name=fname, period=period, deadline=deadline)
        for (tname, C, T, pi) in tasks:
            flow.add_tasks(Task(name=tname, wcet=C, processor=ps[pi]))
        system.add_flows(flow)
    return system


def run(system, limit=1e9):
    HolisticLocalEDFAnalysis(limit_factor=limit, reset=True).apply(system)
    return {t.name: t.wcrt for t in system.tasks}


def two_task_formula():
    """Test the DERIVATION closed form for d_a <= d_b:
       d_a<d_b: R_a = max(C_a, C_a+C_b-(d_b-d_a)); d_a==d_b: R_a = C_a+C_b.
       Also observe d_a>d_b (DERIVATION panel says R_a = C_a+C_b there)."""
    Ca, Cb, T = 2, 3, 10
    bad = 0
    obs_gt = set()
    for da in range(0, 25):
        for db in range(0, 25):
            s = build([("fa", [("a", Ca, T, 0)]), ("fb", [("b", Cb, T, 0)])])
            s["fa"]["a"].deadline = da
            s["fb"]["b"].deadline = db
            r = run(s)
            if da < db:
                pred = max(Ca, Ca + Cb - (db - da))
            elif da == db:
                pred = Ca + Cb
            else:
                obs_gt.add(r["a"])
                continue
            if abs(r["a"] - pred) > 1e-9:
                bad += 1
                if bad <= 10:
                    print(f"  MISMATCH da={da} db={db}: R_a={r['a']} pred={pred}")
    print(f"two_task_formula (d_a<=d_b): mismatches={bad} (over 325 pairs)")
    print(f"  observed R_a for d_a>d_b: {sorted(obs_gt)} (DERIVATION says 5)")


def da_cancellation():
    """Check: at psi = psi_ab the direct +d_a cancels; residual through pd_t."""
    # victim a(C=2,T=10) + interferer t(C=2,T=10); fix d_t large; vary d_a
    print("da at psi_ab (p=1): r = w - 0 ; w depends on d_a via pd_t")
    for da in [0, 1, 2, 3, 4, 5, 8, 12]:
        s = build([("fa", [("a", 2, 10, 0)]), ("ft", [("t", 2, 10, 0)])])
        s["fa"]["a"].deadline = da
        s["ft"]["t"].deadline = 100  # far away: pd_t at psi_ab = d_a?
        r = run(s)
        print(f"   d_a={da:>3d} -> R_a={r['a']:.3f}  R_t={r['t']:.3f}")


def totals(methods, seed=42, population=50):
    report = harness.evaluate(techniques=methods, seed=seed, population=population)
    return {n: report["techniques"][n]["schedulable"]
            for n in report["techniques"]}


def main():
    print("== two-task closed form ==")
    two_task_formula()
    print("\n== d_a cancellation probe ==")
    da_cancellation()

    print("\n== registered technique totals, seed 42 ==")
    from algorithms import load
    reg = load()
    for s in [42, 1, 7, 123]:
        rep = harness.evaluate(seed=s)
        line = "  ".join(f"{n}={rep['techniques'][n]['schedulable']}"
                         for n in ["pd", "pd_global", "wad", "wad_adapt",
                                   "bl_scale", "bl_adaptive"])
        print(f"seed {s:4d}: {line}")


if __name__ == "__main__":
    main()
