"""Deadline statistics of the WAD family.

Size 15 is computed over the full benchmark (all 20 utilisation levels); size 25
is sampled at every fourth level (5 levels) to keep the run short. For each rule:
the minimum d_i/C_i over all tasks, and the mean/maximum of sum_i d_i/D_f per flow.
"""
import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from assignment.wad_adapt_assignment import WADAdaptAssignment
from assignment.wad_assignment import WADAssignment
from examples.generator import to_edf
from sweep import levels
from workspace.framework_paper.systems import get_systems

HERE = os.path.dirname(os.path.abspath(__file__))
UTIL = np.linspace(0.5, 0.9, 20)
RULES = {"wad": WADAssignment,
         "wad_adapt": WADAdaptAssignment,
         "wad_budget": lambda: WADAdaptAssignment(budget=True)}


def stats(size, record_every=1):
    base = get_systems(size)
    for s in base:
        to_edf(s)
    deg = {k: [] for k in RULES}
    over = {k: [] for k in RULES}
    for _, _, pop in levels(base, UTIL, record=lambda i: i % record_every == 0):
        for s0 in pop:
            for name, factory in RULES.items():
                s = deepcopy(s0)
                factory().apply(s)
                for t in s.tasks:
                    deg[name].append(t.deadline / t.wcet)
                for f in s.flows:
                    over[name].append(sum(t.deadline for t in f.tasks) / f.deadline)
    return {name: {"min_d_over_C": round(min(deg[name]), 3),
                   "mean_sumD_over_Df": round(sum(over[name]) / len(over[name]), 2),
                   "max_sumD_over_Df": round(max(over[name]), 2)}
            for name in RULES}


if __name__ == "__main__":
    res = {"15 (full)": stats(15, 1),
           "25 (sampled, 5 levels)": stats(25, 4)}
    print(json.dumps(res, indent=1))
    pd.DataFrame(res).T.to_csv(os.path.join(HERE, "stats_full.csv"))
