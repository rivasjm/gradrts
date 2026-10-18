"""EDF evaluation of the sandbox-derived deadline assignments (WAD family).

Same protocol as ``workspace/framework_paper/edf/edf.py`` (identical pool, sizes
and utilization sweep), but evaluating, under local EDF:

- ``pd``          -- proportional deadlines (baseline, ``PDAssignment``);
- ``wad``         -- work-and-position aware one-pass rule (``WADAssignment``);
- ``wad_adapt``   -- WAD seed + 3 bounded per-flow correction passes
                     (``WADAdaptAssignment``);
- ``wad_budget``  -- the same, projected so that ``sum_i d_i = D_f`` per flow
                     (budget-feasible variant);
- ``rta_cd``      -- response-time-guided coordinate descent
                     (``RTACDAssignment``, size 15 only, as a search reference).

This lives in its own package so the paper's evaluation is left untouched, and
writes its results to its own ``edf-wad-<size>/`` output directories.

Usage (from ``code/``)::

    .venv/bin/python workspace/edf_wad/edf_wad.py        # both sizes
    .venv/bin/python workspace/edf_wad/edf_wad.py 15     # only size 15
"""

import argparse
import os

import numpy as np

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import EQFAssignment, EQSAssignment, PDAssignment
from assignment.rta_cd_assignment import RTACDAssignment
from assignment.wad_adapt_assignment import WADAdaptAssignment
from assignment.wad_assignment import WADAssignment
from examples.evaluation import SchedRatioEval
from examples.generator import to_edf
from workspace.framework_paper.systems import SIZES, get_systems

# utilizations between 50 % and 90 %
UTILIZATIONS = np.linspace(0.5, 0.9, 20)


def item(system, assignment, test):
    assignment.apply(system)
    test.apply(system)
    return system.is_schedulable()


def edf_local_pd(system):
    return item(system, PDAssignment(), HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


def edf_local_wad(system):
    return item(system, WADAssignment(), HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


def edf_local_wad_adapt(system):
    return item(system, WADAdaptAssignment(), HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


def edf_local_wad_budget(system):
    return item(system, WADAdaptAssignment(budget=True), HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


def edf_local_eqf(system):
    return item(system, EQFAssignment(), HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


def edf_local_eqs(system):
    return item(system, EQSAssignment(), HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


def edf_local_rta_cd(system):
    return item(system, RTACDAssignment(), HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


def run_size(size: int, base_output_dir: str) -> None:
    eval_name = f"edf-wad-{size}"
    systems = get_systems(size)
    for system in systems:
        to_edf(system)

    tools = [("pd", edf_local_pd),
             ("wad", edf_local_wad),
             ("wad_adapt", edf_local_wad_adapt),
             ("wad_budget", edf_local_wad_budget),
             ("eqf", edf_local_eqf),
             ("eqs", edf_local_eqs)]
    # rta_cd is a per-instance search: it is the strongest but very slow, so it is
    # only run on the smaller scenario.
    if size == 15:
        tools.append(("rta_cd", edf_local_rta_cd))

    labels, funcs = zip(*tools)
    output_dir = os.path.join(base_output_dir, eval_name)
    os.makedirs(output_dir, exist_ok=True)
    runner = SchedRatioEval(eval_name, labels=labels, funcs=funcs,
                            systems=systems, utilizations=UTILIZATIONS, threads=6,
                            output_dir=output_dir)
    runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="EDF evaluation of the WAD family")
    parser.add_argument("size", type=int, choices=sorted(SIZES), nargs="*", default=None,
                        help="System size(s) (total number of tasks); can be repeated. "
                             "Default: every size")
    parser.add_argument("-o", "--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for generated files (default: script directory)")
    args = parser.parse_args()

    for size in (args.size or sorted(SIZES)):
        run_size(size, args.output_dir)
