"""Validate VectorHolisticEDFAnalysis against scalar HolisticLocalEDFAnalysis."""

import copy
import time
import numpy as np
from random import Random

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from model.linear_system import SchedulerType
from vector.vector_edf import VectorHolisticEDFAnalysis, ResultsCache


def compare_one(system, M_psi=100):
    """Return (scalar_wcrts, vector_wcrts) for the same system."""
    s1 = copy.deepcopy(system)
    scalar = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    scalar.apply(s1)
    sw = np.array([t.wcrt for t in s1.tasks])

    s2 = copy.deepcopy(system)
    vec = VectorHolisticEDFAnalysis(M_psi=M_psi, limit_factor=10, cache=ResultsCache())
    vec.apply(s2)
    vw = np.array([t.wcrt for t in s2.tasks])

    return sw, vw


def main():
    rnd = Random(42)
    n_sys = 20
    systems = [get_system((3, 4, 3), rnd, balanced=True, name=str(i),
                          deadline_factor_min=0.3, sched=SchedulerType.EDF,
                          deadline_factor_max=0.7)
               for i in range(n_sys)]

    configs = [50, 100, 200, 500]

    for M in configs:
        all_scalar = []
        all_vector = []
        errors = []
        times_scalar = []
        times_vec = []

        for sys_ in systems:
            PDAssignment().apply(sys_)

            t0 = time.perf_counter()
            sw, vw = compare_one(sys_, M_psi=M)
            t = time.perf_counter() - t0

            mask = sw > 0
            if mask.sum() > 1:
                err = np.abs(vw[mask] - sw[mask]) / sw[mask]
                errors.extend(err.tolist())
            all_scalar.extend(sw.tolist())
            all_vector.extend(vw.tolist())

        all_scalar = np.array(all_scalar)
        all_vector = np.array(all_vector)
        mask = (all_scalar > 0) & np.isfinite(all_scalar) & np.isfinite(all_vector)
        corr = np.corrcoef(all_scalar[mask], all_vector[mask])[0, 1]
        errs = np.array(errors)

        print(f"M={M:4d}: Pearson r={corr:.4f}  "
              f"rel.err mean={np.mean(errs):.4f} max={np.max(errs):.4f}  "
              f"abs.err mean={np.mean(np.abs(all_vector[mask]-all_scalar[mask])):.2f}")


if __name__ == "__main__":
    main()
