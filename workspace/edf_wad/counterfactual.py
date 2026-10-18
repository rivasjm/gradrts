"""Counterfactual analysis variants for the mechanism discussion.

Three analyses are compared on the EDF-15 pool:
  * full        -- the judge (HolisticLocalEDFAnalysis, eq. (1));
  * cutoff-only -- keep only the cut-off ``D < d_i``, drop the floor cap;
  * no-cap      -- ignore the local deadline entirely in the interference.
For each, sweep the uniform scale lambda and run WAD. Writes counterfactual.csv.
"""
import math
import os
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
LAMBDAS = [0.5, 1.0, 2.0, 3.0, 5.0]


class CutoffOnly(HolisticLocalEDFAnalysis):
    @staticmethod
    def _wi(task, t, D):
        if D < task.deadline:
            return 0.0
        return math.ceil((t + task.jitter) / task.period) * task.wcet


class NoCap(HolisticLocalEDFAnalysis):
    @staticmethod
    def _wi(task, t, D):
        return math.ceil((t + task.jitter) / task.period) * task.wcet


def apply_family(system, lam):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        for t in flow.tasks:
            t.deadline = lam * t.wcet * flow.deadline / total


ANALYSES = [("full", HolisticLocalEDFAnalysis),
            ("cutoff_only", CutoffOnly),
            ("no_cap", NoCap)]
CONFIGS = [(f"lambda={l:g}", l, False) for l in LAMBDAS] + [("wad", None, True)]


def run():
    base = get_systems(15)
    for s in base:
        to_edf(s)
    counts = {(name, cfg): 0 for name, _ in ANALYSES for cfg, _, _ in CONFIGS}
    for _, _, pop in levels(base, UTIL):
        for s0 in pop:
            for name, cls in ANALYSES:
                for cfg, lam, wad in CONFIGS:
                    s = deepcopy(s0)
                    if wad:
                        WADAssignment().apply(s)
                    else:
                        apply_family(s, lam)
                    cls(limit_factor=1, reset=True).apply(s)
                    counts[name, cfg] += s.is_schedulable()
    return counts


if __name__ == "__main__":
    counts = run()
    rows = {name: [counts[name, cfg] for cfg, _, _ in CONFIGS] for name, _ in ANALYSES}
    df = pd.DataFrame(rows, index=[cfg for cfg, _, _ in CONFIGS])
    df.to_csv(os.path.join(HERE, "counterfactual.csv"))
    print(df)
