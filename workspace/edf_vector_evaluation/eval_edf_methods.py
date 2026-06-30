"""Schedulability sweep: compare EDF local optimisation methods.

Methods:
  1. EDF-L PD         — Deadline Monotonic baseline
  2. EDF-L GDPA-Surr  — Gradient descent with differentiable surrogate
  3. EDF-L GDPA-Vec   — Gradient descent with vectorised exact analysis
  4. EDF-L GDPA-Seq   — Gradient descent with sequential finite-differences
"""

import numpy as np
from random import Random

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from examples.evaluation import SchedRatioEval
from examples.example_models import get_system
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import SequentialGradientFunction
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import DeadlineExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem, SchedulerType
from surrogate.surrogate_edf import SurrogateEDFGradient
from vector.vector_edf import VectorEDFGradientFunction


# =========================================================================
# Tool definitions  (module-level for multiprocessing pickling)
# =========================================================================

def edf_pd(system: LinearSystem) -> bool:
    PDAssignment().apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def edf_gdpa_surr(system: LinearSystem) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
    stop_fn = ThresholdStopFunction(limit=200, patience=50)
    grad_fn = SurrogateEDFGradient(tau=0.05, N_w=10, N_jitter=2, M_psi=50,
                                   temperature_max=0.1, grad_clip=1.0)
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
    grad_fn = VectorEDFGradientFunction(sigma=1.5, M_psi=100, cost_limit_factor=10)
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


def edf_gdpa_seq(system: LinearSystem) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
    stop_fn = ThresholdStopFunction(limit=50, patience=15)
    grad_fn = SequentialGradientFunction(cost_function=cost_fn, sigma=1.5)
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
    threads = 1  # torch (surrogate) is not fork-safe

    print("=" * 60)
    print("EDF Local — Method Comparison")
    print(f"  Systems: {n_systems},  size: {size}")
    print(f"  Utils: {n_utils} levels")
    print(f"  Threads: {threads}")
    print("=" * 60)

    systems = [
        get_system(size, rnd, balanced=True, name=str(i),
                   deadline_factor_min=0.3, sched=SchedulerType.EDF,
                   deadline_factor_max=0.7)
        for i in range(n_systems)
    ]
    utilizations = np.linspace(0.5, 0.9, n_utils)

    tools = [
        ("EDF-L PD",         edf_pd),
        ("EDF-L GDPA-Surr",  edf_gdpa_surr),
        ("EDF-L GDPA-Vec",   edf_gdpa_vec),
    ]

    labels, funcs = zip(*tools)

    runner = SchedRatioEval(
        name="edf_vector_eval_full",
        labels=labels,
        funcs=funcs,
        systems=systems,
        utilizations=utilizations,
        threads=threads,
        output_dir="workspace/edf_vector_evaluation",
    )
    runner.run()
    print("\nDone — results in workspace/edf_vector_evaluation/")

