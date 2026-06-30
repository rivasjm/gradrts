"""Full schedulability sweep: all EDF local methods.

Methods:
  1. EDF-L PD         — Deadline Monotonic baseline
  2. EDF-L HOPA       — Heuristic Optimised Priority Assignment
  3. EDF-L GDPA-Surr  — Gradient descent (differentiable surrogate gradient)
  4. EDF-L GDPA-Vec   — Gradient descent (vectorised exact gradient)
  5. EDF-L GDPA-Seq   — Gradient descent (sequential FD gradient, gold standard)

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
from gradient_descent.gradient_function import SequentialGradientFunction
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import DeadlineExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem, SchedulerType
from surrogate.surrogate_edf import SurrogateEDFGradient
from vector.vector_edf import VectorEDFGradientFunction


# =========================================================================
# Tool definitions
# =========================================================================

def edf_pd(system: LinearSystem) -> bool:
    """Deadline Monotonic baseline."""
    PDAssignment().apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def edf_hopa(system: LinearSystem) -> bool:
    """HOPA iterative heuristic."""
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    hopa = HOPAssignment(analysis=analysis)
    hopa.apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def edf_gdpa_surr(system: LinearSystem) -> bool:
    """GDPA with differentiable surrogate gradient (200 iter, tau=0.05)."""
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
    """GDPA with vectorised exact gradient (50 iter, M_psi=100)."""
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
    """GDPA with sequential FD gradient (50 iter) — gold standard."""
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
    size = (3, 4, 3)       # flows, tasks/flow, processors
    n_systems = 20          # population size
    n_utils = 12            # utilisation levels (0.5 … 0.9)
    threads = 1             # torch (surrogate) not fork-safe

    print("=" * 65)
    print("EDF Local — Full Method Comparison")
    print(f"  Systems: {n_systems},  size: {size}")
    print(f"  Utils: {n_utils}  |  Threads: {threads}")
    print("=" * 65)
    print(f"  Methods:")
    print(f"    1. EDF-L PD         — Deadline Monotonic baseline")
    print(f"    2. EDF-L HOPA       — Heuristic Optimised Priority Assignment")
    print(f"    3. EDF-L GDPA-Surr  — GD + differentiable surrogate  (200 iter)")
    print(f"    4. EDF-L GDPA-Vec   — GD + vectorised exact analysis (50 iter)")
    print(f"    5. EDF-L GDPA-Seq   — GD + sequential FD            (50 iter) [SLOW]")
    print(f"  Estimated runtime: ~25-40 min with Seq, ~8-12 min without")

    # Generate population
    systems = [
        get_system(size, rnd, balanced=True, name=str(i),
                   deadline_factor_min=0.3, sched=SchedulerType.EDF,
                   deadline_factor_max=0.7)
        for i in range(n_systems)
    ]
    utilizations = np.linspace(0.5, 0.9, n_utils)

    # ---- Choose which methods to run ----
    # Toggle GDPA-Seq: set to None to skip (saves ~60% runtime)
    include_seq = True  # change to True to include the gold-standard

    all_tools = [
        ("EDF-L PD",         edf_pd),
        ("EDF-L HOPA",       edf_hopa),
        ("EDF-L GDPA-Surr",  edf_gdpa_surr),
        ("EDF-L GDPA-Vec",   edf_gdpa_vec),
    ]
    if include_seq:
        all_tools.append(("EDF-L GDPA-Seq", edf_gdpa_seq))

    labels, funcs = zip(*all_tools)

    runner = SchedRatioEval(
        name="edf_all_methods",
        labels=labels,
        funcs=funcs,
        systems=systems,
        utilizations=utilizations,
        threads=threads,
        output_dir=os.path.dirname(os.path.abspath(__file__)),
    )
    runner.run()

    print("\nDone — results in workspace/edf_vector_evaluation/edf_all_methods_*")
