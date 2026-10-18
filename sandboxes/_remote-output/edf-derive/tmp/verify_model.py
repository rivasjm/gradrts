"""Verifier: does the rule exploit the model by setting d_i below C_i (or <=0)?"""

import copy
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from algorithms import load  # noqa: E402
from frozen import harness, pool  # noqa: E402


def main():
    techs = load()
    systems = pool.get_systems()
    for name in ["wad", "wad_adapt", "bl_scale", "bl_adaptive", "pd", "pd_global"]:
        f = techs[name]
        n_below = 0
        n_inst_with_below = 0
        n_sched = 0
        n_sched_below = 0
        worst = 1e9
        for u in pool.UTILIZATIONS:
            for base in systems:
                work = copy.deepcopy(base)
                pool.set_utilization(work, u)
                s = copy.deepcopy(work)
                f(s)
                below = sum(1 for t in s.tasks if t.deadline < t.wcet)
                r = min(t.deadline / t.wcet for t in s.tasks)
                worst = min(worst, r)
                n_below += below
                if below:
                    n_inst_with_below += 1
                if harness.is_schedulable(s):
                    n_sched += 1
                    if below:
                        n_sched_below += 1
        print(f"{name:12s} tasks d<C: {n_below:5d}/15000  instances: "
              f"{n_inst_with_below:4d}/1000  schedulable: {n_sched:4d}  "
              f"of which have a d<C task: {n_sched_below:4d}  min(d/C)={worst:.3g}")


if __name__ == "__main__":
    main()
