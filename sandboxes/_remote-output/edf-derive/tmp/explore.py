"""Scratch probe: explore the analysis on tiny systems.

Run with the repo venv:
    ../../../.venv/bin/python tmp/explore.py
"""

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen import explain, format_detail  # noqa: E402
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402
from frozen.model import (Flow, LinearSystem, Processor, SchedulerType,  # noqa: E402
                          Task)


def make_proc_system(specs):
    """specs: list of (name, wcet, period, deadline, proc_index).

    Each spec becomes its own single-task flow (no jitter) so we can study
    pure single-processor interference.
    """
    system = LinearSystem()
    procs = [Processor(name=f"p{i}", sched=SchedulerType.EDF) for i in range(3)]
    system.add_procs(*procs)
    for (name, C, T, D, pi) in specs:
        flow = Flow(name=f"f_{name}", period=T, deadline=D)
        task = Task(name=name, wcet=C, processor=procs[pi])
        flow.add_tasks(task)
        system.add_flows(flow)
    return system


def wcrt_of(system, limit=1000):
    HolisticLocalEDFAnalysis(limit_factor=limit, reset=True).apply(system)
    return {t.name: t.wcrt for t in system.tasks}


def sweep_two_task():
    print("=== Sweep d1,d2 on one processor, two independent tasks ===")
    # task a: C=2, T=10 ; task b: C=3, T=10
    for d1 in [2, 4, 6, 8, 10, 15, 20, 30]:
        row = []
        for d2 in [2, 4, 6, 8, 10, 15, 20, 30]:
            s = make_proc_system([("a", 2, 10, 100, 0), ("b", 3, 10, 100, 0)])
            s["f_a"].tasks[0].deadline = d1
            s["f_b"].tasks[0].deadline = d2
            r = wcrt_of(s)
            row.append(f"({r['a']:.1f},{r['b']:.1f})")
        print(f" d1={d1:>3}: " + " ".join(f"{x:>9}" for x in row))
    print(" (columns are d2 = 2,4,6,8,10,15,20,30)")


def detail_two_task(d1, d2):
    s = make_proc_system([("a", 2, 10, 100, 0), ("b", 3, 10, 100, 0)])
    s["f_a"].tasks[0].deadline = d1
    s["f_b"].tasks[0].deadline = d2
    ok, details = explain(s, limit_factor=1000)
    print(f"--- d1={d1} d2={d2} ok={ok}")
    for n in ("a", "b"):
        print(format_detail(n, details[n]))


if __name__ == "__main__":
    sweep_two_task()
    print()
    detail_two_task(2, 20)
    detail_two_task(8, 20)
    detail_two_task(20, 20)
