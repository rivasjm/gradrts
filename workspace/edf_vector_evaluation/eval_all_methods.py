"""Full schedulability sweep: all EDF local methods.

Methods:
  1. EDF-L PD         — Deadline Monotonic baseline
  2. EDF-L HOPA       — Heuristic Optimised Priority Assignment
  3. EDF-L GDPA-Surr  — Gradient descent (differentiable surrogate)
  4. EDF-L GDPA-Vec   — Gradient descent (vectorised V2, exact psi set)

Run from code/ with:
    PYTHONPATH=. python3 workspace/edf_vector_evaluation/eval_all_methods.py
"""

import os
import numpy as np
from random import Random

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from assignment.hopa_assignment import HOPAssignment
from examples.evaluation import SchedRatioEval
from examples.example_models import get_system
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import DeadlineExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem, SchedulerType
from surrogate.surrogate_edf import SurrogateEDFGradient
from vector.vector_edf_v2 import VectorEDFGradientFunctionV2


# =========================================================================
# Tool definitions
# =========================================================================

def edf_pd(system: LinearSystem) -> bool:
    PDAssignment().apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def edf_hopa(system: LinearSystem) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    hopa = HOPAssignment(analysis=analysis)
    hopa.apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def edf_gdpa_surr(system: LinearSystem) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
    stop_fn = ThresholdStopFunction(limit=200, patience=50)
    grad_fn = SurrogateEDFGradient(
        tau=0.05, N_w=10, N_jitter=2, M_psi=50,
        temperature_max=0.1, grad_clip=1.0,
    )
    update_fn = NoisyAdam()
    optimizer = GradientDescentOptimizer(
        parameter_handler=ph, cost_function=cost_fn,
        stop_function=stop_fn, gradient_function=grad_fn,
        update_function=update_fn, verbose=False,
    )
    PDAssignment().apply(system)
    optimizer.apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def edf_gdpa_vec(system: LinearSystem) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
    stop_fn = ThresholdStopFunction(limit=50, patience=15)
    grad_fn = VectorEDFGradientFunctionV2(sigma=1.5, cost_limit_factor=10)
    update_fn = NoisyAdam()
    optimizer = GradientDescentOptimizer(
        parameter_handler=ph, cost_function=cost_fn,
        stop_function=stop_fn, gradient_function=grad_fn,
        update_function=update_fn, verbose=False,
    )
    PDAssignment().apply(system)
    optimizer.apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    rnd = Random(42)
    size = (3, 4, 3)
    n_systems = 20
    n_utils = 12
    threads = 1  # torch (surrogate) not fork-safe

    print("=" * 65)
    print("EDF Local — Method Comparison (V2)")
    print(f"  Systems: {n_systems},  size: {size}")
    print(f"  Utils: {n_utils}  |  Threads: {threads}")
    print("=" * 65)
    print(f"  Methods:")
    print(f"    1. EDF-L PD         — Deadline Monotonic baseline")
    print(f"    2. EDF-L HOPA       — Heuristic Optimised Priority Assignment")
    print(f"    3. EDF-L GDPA-Surr  — GD + differentiable surrogate  (200 iter)")
    print(f"    4. EDF-L GDPA-Vec   — GD + vectorised V2 exact       (50 iter)")

    systems = [
        get_system(size, rnd, balanced=True, name=str(i),
                   deadline_factor_min=0.3, sched=SchedulerType.EDF,
                   deadline_factor_max=0.7)
        for i in range(n_systems)
    ]
    utilizations = np.linspace(0.5, 0.9, n_utils)

    tools = [
        ("EDF-L PD",         edf_pd),
        ("EDF-L HOPA",       edf_hopa),
        ("EDF-L GDPA-Surr",  edf_gdpa_surr),
        ("EDF-L GDPA-Vec",   edf_gdpa_vec),
    ]

    labels, funcs = zip(*tools)

    runner = SchedRatioEval(
        name="edf_v2_methods",
        labels=labels,
        funcs=funcs,
        systems=systems,
        utilizations=utilizations,
        threads=threads,
        output_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    runner.run()

    print("\nDone — results in workspace/edf_vector_evaluation/edf_v2_methods_*")
