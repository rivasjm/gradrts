"""Benchmark: compare schedulability ratios and run-times of:
   1. PD           (Deadline Monotonic baseline)
   2. GDPA-Seq     (Gradient descent with SequentialGradientFunction — finite differences)
   3. GDPA-Surrogate (Gradient descent with SurrogateEDFGradient — differentiable surrogate)
"""

import time
from random import Random
import numpy as np

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from examples.evaluation import SchedRatioEval
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import SequentialGradientFunction
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import DeadlineExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem, SchedulerType
from surrogate.surrogate_edf import SurrogateEDFGradient


# =========================================================================
# Tool definitions
# =========================================================================

def edf_pd(system: LinearSystem) -> bool:
    """PD (Deadline Monotonic) baseline."""
    PDAssignment().apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def edf_gdpa_sequential(system: LinearSystem) -> bool:
    """GDPA with finite-difference gradient (gold-standard, slow)."""
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    parameter_handler = DeadlineExtractor()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = SequentialGradientFunction(cost_function=cost_function, sigma=1.5)
    update_function = NoisyAdam()

    optimizer = GradientDescentOptimizer(
        parameter_handler=parameter_handler,
        cost_function=cost_function,
        stop_function=stop_function,
        gradient_function=gradient_function,
        update_function=update_function,
        verbose=False,
    )

    PDAssignment().apply(system)
    optimizer.apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def edf_gdpa_surrogate(system: LinearSystem, tau=0.5, N_w=10) -> bool:
    """GDPA with differentiable EDF surrogate gradient."""
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    parameter_handler = DeadlineExtractor()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = SurrogateEDFGradient(
        tau=tau, N_w=N_w, N_jitter=2, M_psi=50,
        temperature_max=0.1, grad_clip=1.0,
    )
    update_function = NoisyAdam()

    optimizer = GradientDescentOptimizer(
        parameter_handler=parameter_handler,
        cost_function=cost_function,
        stop_function=stop_function,
        gradient_function=gradient_function,
        update_function=update_function,
        verbose=False,
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
    size = (3, 4, 3)     # (flows, tasks/flow, processors)
    n_systems = 50

    print("=" * 60)
    print("EDF Surrogate Benchmark")
    print(f"  Systems: {n_systems},  size: {size}")
    print("=" * 60)

    # Generate population
    systems = [
        get_system(size, rnd, balanced=True, name=str(i),
                   deadline_factor_min=0.3, sched=SchedulerType.EDF,
                   deadline_factor_max=0.7)
        for i in range(n_systems)
    ]

    utilizations = np.linspace(0.5, 0.9, 20)

    # ----- Full schedulability sweep -----
    tools = [
        ("EDF-L PD", edf_pd),
        ("EDF-L GDPA-Seq", edf_gdpa_sequential),
        ("EDF-L GDPA-Surr", edf_gdpa_surrogate),
    ]

    print("\n--- Schedulability sweep ---")
    labels, funcs = zip(*tools)
    runner = SchedRatioEval(
        "edf_surrogate_benchmark",
        labels=labels,
        funcs=funcs,
        systems=systems,
        utilizations=utilizations,
        threads=1,  # sequential to avoid torch multiprocessing issues
    )
    runner.run()
    print("  → results saved.")

    # ----- Timing comparison -----
    print("\n--- Timing comparison (single system, single run) ---")
    test_sys = systems[0]
    PDAssignment().apply(test_sys)

    # Warm-up
    HolisticLocalEDFAnalysis(limit_factor=10, reset=False).apply(test_sys)

    # Time sequential gradient
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
    seq_grad = SequentialGradientFunction(cost_function=cost_fn, sigma=1.5)
    x = ph.extract(test_sys)

    t0 = time.perf_counter()
    _ = seq_grad.compute(test_sys, x)
    t_seq = time.perf_counter() - t0
    print(f"  SequentialGradientFunction:  {t_seq:.3f} s")

    # Time surrogate gradient
    surr_grad = SurrogateEDFGradient(tau=0.5, N_w=10, N_jitter=2, M_psi=50)
    t0 = time.perf_counter()
    _ = surr_grad.compute(test_sys, x)
    t_surr = time.perf_counter() - t0
    print(f"  SurrogateEDFGradient:        {t_surr:.3f} s")
    print(f"  Speed-up:                    {t_seq / t_surr:.1f}x")

    # Time single analysis (for reference)
    t0 = time.perf_counter()
    analysis.apply(test_sys)
    t_ana = time.perf_counter() - t0
    print(f"  HolisticLocalEDFAnalysis:    {t_ana:.6f} s  (×{len(x)*2} per seq-grad = {t_ana*len(x)*2:.3f} s)")

    print("\nDone.")
