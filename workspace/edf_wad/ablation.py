"""Order-versus-magnitude ablation on the EDF-15 pool (project protocol).

* Permutation: take the WAD deadlines and, per instance, shuffle each flow's
  deadlines among its tasks (20 random shuffles), counting how often a shuffle
  is schedulable where the identity is not.
* Grid: the two-lever family d_k = lambda * (s_k + gamma * sum_{j<k} s_j) over a
  grid of (gamma, lambda).
Writes ablation.csv (summary).

Uses the same population sweep as the main runner (``sweep.levels``).
"""
import os
import random
from copy import deepcopy

import numpy as np
import pandas as pd

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.wad_assignment import WADAssignment
from examples.generator import to_edf
from sweep import levels
from workspace.framework_paper.systems import get_systems

HERE = os.path.dirname(os.path.abspath(__file__))
UTIL = np.linspace(0.5, 0.9, 20)
GAMMAS = [0.0, 0.2, 0.3, 0.5, 0.6, 1.0]
LAMBDAS = [1.0, 1.5, 2.0, 3.0]
SHUFFLES = 20


def sched(system):
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def apply_grid(system, gamma, lam):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        prev = 0.0
        for t in flow.tasks:
            share = t.wcet * flow.deadline / total
            t.deadline = lam * (share + gamma * prev)
            prev += share


if __name__ == "__main__":
    base = get_systems(15)
    for s in base:
        to_edf(s)

    rng = random.Random(0)
    identity = shuffled_best = flat_improve = flat_equal = flat_worse = 0
    grid = {(g, l): 0 for g in GAMMAS for l in LAMBDAS}
    for _, _, pop in levels(base, UTIL):
        for s0 in pop:
            s = deepcopy(s0)
            WADAssignment().apply(s)
            id_ok = sched(s)
            identity += id_ok
            any_better = False
            for _ in range(SHUFFLES):
                cand = deepcopy(s)
                for flow in cand.flows:
                    ds = [t.deadline for t in flow.tasks]
                    rng.shuffle(ds)
                    for t, d in zip(flow.tasks, ds):
                        t.deadline = d
                ok = sched(cand)
                any_better = any_better or ok
                if ok and not id_ok:
                    flat_improve += 1
                elif ok == id_ok:
                    flat_equal += 1
                else:
                    flat_worse += 1
            if any_better and not id_ok:
                shuffled_best += 1

            for g in GAMMAS:
                for l in LAMBDAS:
                    c = deepcopy(s0)
                    apply_grid(c, g, l)
                    grid[(g, l)] += sched(c)

    rows = {f"g={g:g}": [grid[(g, l)] for l in LAMBDAS] for g in GAMMAS}
    table = pd.DataFrame(rows, index=[f"lambda={l:g}" for l in LAMBDAS])
    table.to_csv(os.path.join(HERE, "ablation_grid.csv"))
    summary = {"identity_schedulable": identity,
               "instances_where_a_shuffle_helps": shuffled_best,
               "shuffles_schedulable_where_identity_not": flat_improve,
               "shuffles_equal": flat_equal,
               "shuffles_worse": flat_worse,
               "total_shuffles": 1000 * SHUFFLES}
    pd.Series(summary).to_csv(os.path.join(HERE, "ablation.csv"))
    print(table)
    print(summary)
