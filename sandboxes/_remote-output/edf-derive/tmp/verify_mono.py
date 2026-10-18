"""Verifier: is R_y globally monotone decreasing in another task's deadline d_x?

The DERIVATION derives *pointwise* direction (pd weakly decreasing in d_x at a
fixed window point) and then treats it as a global lever.  Search for a case
where increasing d_x *increases* a processor-mate's response.
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from frozen.analysis import HolisticLocalEDFAnalysis  # noqa: E402
from frozen.model import (Flow, LinearSystem, Processor,  # noqa: E402
                          SchedulerType, Task)


def build(specs, T=10):
    system = LinearSystem()
    p = Processor(name="p0", sched=SchedulerType.EDF)
    system.add_procs(p)
    for i, C in enumerate(specs):
        f = Flow(name=f"f{i}", period=T, deadline=1e9)
        f.add_tasks(Task(name=f"t{i}", wcet=C, processor=p))
        system.add_flows(f)
    return system


def responses(CS, deadlines):
    s = build([c for c in CS])
    for t, d in zip(s.tasks, deadlines):
        t.deadline = d
    HolisticLocalEDFAnalysis(limit_factor=1e9, reset=True).apply(s)
    return [t.wcrt for t in s.tasks]


def main():
    rnd = random.Random(0)
    nonmono = 0
    checked = 0
    examples = []
    for trial in range(4000):
        k = rnd.randint(2, 4)
        CS = [round(rnd.uniform(1, 5), 2) for _ in range(k)]
        if sum(CS) / 10.0 >= 0.98:
            continue
        others = [rnd.choice([0, 1, 2, 3, 5, 9, 15]) for _ in range(k)]
        # sweep d_x for each x, look at every y != x
        for x in range(k):
            prev = None
            for dx in range(0, 41):
                ds = list(others)
                ds[x] = dx
                r = responses(CS, ds)
                if prev is not None:
                    for y in range(k):
                        if y == x:
                            continue
                        if r[y] > prev[y] + 1e-9:
                            nonmono += 1
                            if len(examples) < 8:
                                examples.append((CS, others, x, dx, y,
                                                 prev[y], r[y]))
                prev = r
            checked += 1
    print(f"non-monotone increases found: {nonmono} (over {checked} sweeps)")
    for CS, others, x, dx, y, a, b in examples:
        print(f"  C={CS} base={others}: raising d_t{x} to {dx} RAISED R_t{y} "
              f"{a}->{b}")


if __name__ == "__main__":
    main()
