"""Generate the Fig. 1 family curves (EDF-15) and write ``families.csv``.

The two one-parameter families of the observation:
  * partial globalisation  d_i = s_i + gamma * sum_{j<i} s_j   (gamma = 0..1),
  * uniform scale          d_i = lambda * s_i                   (lambda list),
with s_i = C_i D_f / sum_C the proportional shares.
Tidy output: columns ``series,param,count`` (schedulable / 1000).

Uses the same population sweep as the main runner (``sweep.levels``), so the
lambda family in this figure and in the counterfactual table agree.
"""
import os
from copy import deepcopy

import numpy as np
import pandas as pd

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from examples.generator import to_edf
from sweep import levels
from workspace.framework_paper.systems import get_systems

HERE = os.path.dirname(os.path.abspath(__file__))
UTIL = np.linspace(0.5, 0.9, 20)
GAMMAS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAMBDAS = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]


def apply_family(system, gamma=None, lam=None):
    for flow in system.flows:
        total = sum(t.wcet for t in flow.tasks)
        prev = 0.0
        for t in flow.tasks:
            share = t.wcet * flow.deadline / total
            d = share + (gamma * prev if gamma is not None else 0.0)
            t.deadline = d if lam is None else lam * share
            prev += share


if __name__ == "__main__":
    base = get_systems(15)
    for s in base:
        to_edf(s)
    counts = {}
    for _, _, pop in levels(base, UTIL):
        for s0 in pop:
            for g in GAMMAS:
                s = deepcopy(s0)
                apply_family(s, gamma=g)
                HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(s)
                counts[("gamma", g)] = counts.get(("gamma", g), 0) + s.is_schedulable()
            for lam in LAMBDAS:
                s = deepcopy(s0)
                apply_family(s, lam=lam)
                HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(s)
                counts[("lambda", lam)] = counts.get(("lambda", lam), 0) + s.is_schedulable()
    rows = [{"series": series, "param": param, "count": count}
            for (series, param), count in counts.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "families.csv"), index=False)
    print(df.to_string(index=False))
