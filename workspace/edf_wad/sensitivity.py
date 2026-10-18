"""Q x S sensitivity of the one-pass WAD shape (EDF-15). Writes qs_grid.csv.

Uses the same population sweep as the main runner (``sweep.levels``).
"""
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
QS = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
SS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

if __name__ == "__main__":
    base = get_systems(15)
    for s in base:
        to_edf(s)
    grid = np.zeros((len(QS), len(SS)), dtype=int)
    for _, _, pop in levels(base, UTIL):
        for s0 in pop:
            for i, q in enumerate(QS):
                for j, s_frac in enumerate(SS):
                    s = deepcopy(s0)
                    WADAssignment(q=q, s=s_frac).apply(s)
                    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(s)
                    grid[i, j] += s.is_schedulable()
    df = pd.DataFrame(grid, index=QS, columns=SS)
    df.to_csv(os.path.join(HERE, "qs_grid.csv"))
    print(df)
