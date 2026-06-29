"""Benchmark VectorLinearizedV1Analysis vs VectorHolisticFPAnalysis.

For each batch size we run:
  vectorised V1:  bf: VectorLinearizedV1Analysis.apply(system, scenarios)
  vectorised Hol: VectorHolisticFPAnalysis.apply(system, scenarios)

and report wall-clock time and the speed-up of V1 over Holistic per
evaluation.  Also estimates the cost-quality trade-off by checking the
Spearman correlation between V1 and Holistic on the same random scenarios
for each system.
"""
import time
from random import Random

import numpy as np

try:
    from scipy.stats import spearmanr
except ImportError:
    spearmanr = None

from examples.example_models import get_system
from model.analysis_function import init_wcrt, reset_wcrt
from model.linear_system import SchedulerType
from assignment.assignments import PDAssignment
from vector.vector_fp import VectorHolisticFPAnalysis, PrioritiesMatrix

from workspace.holistic_linearization.vector_v1 import VectorLinearizedV1Analysis


def main(seed=42, n_systems=30, size=(3, 4, 3)):
    rnd = Random(seed)
    systems = [
        get_system(size, rnd, name=str(i),
                   deadline_factor_min=0.5, deadline_factor_max=1,
                   sched=SchedulerType.FP, balanced=True)
        for i in range(n_systems)
    ]
    for s in systems:
        PDAssignment(normalize=True).apply(s)

    rng = np.random.default_rng(123)
    builder = PrioritiesMatrix()

    for bsz in (8, 16, 24, 48, 96):
        # build scenarios
        scen_inputs_per_system = []
        for s in systems:
            base_x = [float(t.priority) for t in s.tasks]
            scen_inputs = [list(np.array(base_x) + rng.normal(0, 0.05, size=len(base_x)))
                           for _ in range(bsz)]
            scen_inputs_per_system.append(scen_inputs)

        # Vector V1
        t0 = time.perf_counter()
        v1_results = []
        for s, scin in zip(systems, scen_inputs_per_system):
            scen_pm = builder.apply(s, scin)
            reset_wcrt(s); init_wcrt(s)
            v1 = VectorLinearizedV1Analysis(limit_factor=10, max_p=1000)
            v1.apply(s, scenarios=scen_pm)
            v1_results.append(v1.scenarios_response_times.copy())
        t_v1 = time.perf_counter() - t0

        # Vector Holistic
        t0 = time.perf_counter()
        hol_results = []
        for s, scin in zip(systems, scen_inputs_per_system):
            scen_pm = builder.apply(s, scin)
            reset_wcrt(s); init_wcrt(s)
            hol = VectorHolisticFPAnalysis(limit_factor=10)
            hol.apply(s, scenarios=scen_pm)
            hol_results.append(hol.scenarios_response_times.copy())
        t_hol = time.perf_counter() - t0

        # Spearman between V1 and Holistic response times, pooled across
        # all scenarios of all systems
        v1_flat = np.concatenate(v1_results, axis=1).ravel()
        hol_flat = np.concatenate(hol_results, axis=1).ravel()
        if spearmanr is not None:
            rho, _ = spearmanr(v1_flat, hol_flat)
        else:
            rho = float(np.corrcoef(v1_flat, hol_flat)[0, 1])

        print(f"batch={bsz:3d}  V1={t_v1:7.2f}s  Holistic={t_hol:7.2f}s  "
              f"V1/Hol speedup={t_hol/max(t_v1,1e-9):5.2f}x  "
              f"spearman={rho:.4f}  "
              f"evals={n_systems*bsz}")


if __name__ == "__main__":
    main()