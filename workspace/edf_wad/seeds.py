"""Seed robustness of the cheap methods (EDF-15). Writes seeds.csv."""
import os
from copy import deepcopy
from random import Random

import numpy as np
import pandas as pd

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from assignment.wad_adapt_assignment import WADAdaptAssignment
from assignment.wad_assignment import WADAssignment
from examples.example_models import get_system
from examples.generator import to_edf
from sweep import levels
from workspace.framework_paper.systems import (DEADLINE_FACTOR_MAX,
                                               DEADLINE_FACTOR_MIN, PERIOD_MAX,
                                               PERIOD_MIN, POPULATION, SIZES)

HERE = os.path.dirname(os.path.abspath(__file__))
UTIL = np.linspace(0.5, 0.9, 20)
SEEDS = [1, 7, 42, 123]


def population(seed, size=15):
    rnd = Random(seed)
    return [get_system(SIZES[size], rnd, balanced=True, name=str(i),
                       deadline_factor_min=DEADLINE_FACTOR_MIN,
                       deadline_factor_max=DEADLINE_FACTOR_MAX,
                       period_min=PERIOD_MIN, period_max=PERIOD_MAX)
            for i in range(POPULATION)]


TECHS = {"pd": PDAssignment, "wad": WADAssignment,
         "wad_adapt": WADAdaptAssignment, "wad_budget": lambda: WADAdaptAssignment(budget=True)}

if __name__ == "__main__":
    rows = {}
    for seed in SEEDS:
        base = population(seed)
        for s in base:
            to_edf(s)
        counts = {k: 0 for k in TECHS}
        for _, _, pop in levels(base, UTIL):
            for s0 in pop:
                for k, factory in TECHS.items():
                    s = deepcopy(s0)
                    factory().apply(s)
                    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(s)
                    counts[k] += s.is_schedulable()
        rows[seed] = counts
        print(seed, counts)
    pd.DataFrame(rows).T.to_csv(os.path.join(HERE, "seeds.csv"))
