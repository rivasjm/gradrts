"""Self-contained HOSPA (HOPA adapted to EDF) re-run on EDF-15 and EDF-25.

The paper used to reuse HOSPA results from the companion study. This script
recomputes them in this repository's own protocol so the table is self-contained.
PD is included as a pool sanity check.

Usage (from ``code/``): .venv/bin/python workspace/edf_wad/hospa.py [size ...]
"""
import os
import sys

import numpy as np

from examples.evaluation import SchedRatioEval
from examples.generator import to_edf
from workspace.framework_paper.edf.edf import edf_local_hopa, edf_local_pd
from workspace.framework_paper.systems import get_systems

HERE = os.path.dirname(os.path.abspath(__file__))
UTIL = np.linspace(0.5, 0.9, 20)
THREADS = 6


def run_size(size):
    systems = get_systems(size)
    for s in systems:
        to_edf(s)
    out = os.path.join(HERE, f"hospa-{size}")
    os.makedirs(out, exist_ok=True)
    runner = SchedRatioEval(f"hospa-{size}", labels=["pd", "hospa"],
                            funcs=[edf_local_pd, edf_local_hopa],
                            systems=systems, utilizations=UTIL,
                            threads=THREADS, output_dir=out)
    runner.run()


if __name__ == "__main__":
    sizes = [int(a) for a in sys.argv[1:]] or [15, 25]
    for size in sizes:
        run_size(size)
