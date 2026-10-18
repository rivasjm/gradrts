"""Mechanism probes for DERIVATION.md (scratch).

    1. eq(1) direction: _wi is weakly decreasing in the interferer deadline d_i.
    2. d_a cancellation: for a psi_ab candidate, r_ab = w - (p-1)T_a + j_a.
    3. one processor, two tasks: response vs the two deadlines (square symmetry).
    4. a two-task chain (jitter): the effect survives holistic coupling.
"""
import os, sys
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402
from frozen.model import Flow, LinearSystem, Processor, SchedulerType, Task  # noqa: E402

A = HolisticLocalEDFAnalysis


def two_task(d1, d2, C1=2, C2=3, T=10):
    s = LinearSystem()
    p = Processor(name="p", sched=SchedulerType.EDF)
    s.add_procs(p)
    f1 = Flow("f1", T, 100); f1.add_tasks(Task("a", C1, p, deadline=d1)); s.add_flows(f1)
    f2 = Flow("f2", T, 100); f2.add_tasks(Task("b", C2, p, deadline=d2)); s.add_flows(f2)
    A(limit_factor=1e9, reset=True).apply(s)
    return s["f1"].wcrt, s["f2"].wcrt


def wi_direction():
    print("== eq(1) _wi(t=5, D=10) for an interferer (C=3, T=10, j=0) ==")
    t = Task("i", 3)
    Flow("fi", 10, 100).add_tasks(t)
    print("   d_i :", [round(A._wi(t, 5.0, 10.0), 2) for t.deadline in (0, 1, 5, 9, 10, 11, 50)])


def two_task_table():
    print("== R_a, R_b on one processor (C_a=2, C_b=3, T=10, D=100) ==")
    cols = (2, 4, 6, 8, 10, 20)
    print(" d1\\d2 " + "".join(f"{c:>14}" for c in cols))
    for d1 in cols:
        row = f"{d1:>6} "
        for d2 in cols:
            ra, rb = two_task(d1, d2)
            row += f"{'(' + format(ra, '.1f') + ',' + format(rb, '.1f') + ')':>14}"
        print(row)


def chain():
    print("== 2-task chain: t0 on p0 -> t1 on p1; an interferer on p1 shares t1's proc ==")
    def build(d0, d1, dinter, C0=2, C1=2, Ci=3, T=10):
        s = LinearSystem()
        p0 = Processor(name="p0", sched=SchedulerType.EDF)
        p1 = Processor(name="p1", sched=SchedulerType.EDF)
        s.add_procs(p0, p1)
        f = Flow("f", T, 100)
        f.add_tasks(Task("t0", C0, p0, deadline=d0), Task("t1", C1, p1, deadline=d1))
        s.add_flows(f)
        g = Flow("g", T, 100)
        g.add_tasks(Task("u", Ci, p1, deadline=dinter))
        s.add_flows(g)
        return s
    for dinter in (2, 5, 10, 20, 50):
        s = build(2, 8, dinter)
        A(limit_factor=1e9, reset=True).apply(s)
        print(f"  d_u={dinter:>3}: R_t0={s['f'].tasks[0].wcrt:5.1f}  "
              f"R_end2end={s['f'].wcrt:6.1f}  (j_t1={s['f'].tasks[1].jitter:4.1f})")


if __name__ == "__main__":
    wi_direction()
    two_task_table()
    chain()
