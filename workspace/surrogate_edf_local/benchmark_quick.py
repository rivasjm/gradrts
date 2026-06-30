"""Quick benchmark: PD vs GDPA-Surrogate on reduced population."""

import time
from random import Random
import numpy as np

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import SequentialGradientFunction
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import DeadlineExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem, SchedulerType
from surrogate.surrogate_edf import SurrogateEDFGradient


def edf_pd(system: LinearSystem) -> bool:
    PDAssignment().apply(system)
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def edf_gdpa_surrogate(system: LinearSystem) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    parameter_handler = DeadlineExtractor()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = SurrogateEDFGradient(
        tau=0.5, N_w=10, N_jitter=2, M_psi=50,
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


def edf_gdpa_sequential(system: LinearSystem) -> bool:
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


if __name__ == "__main__":
    rnd = Random(42)
    size = (3, 4, 3)
    n_systems = 10
    n_utils = 10

    systems = [
        get_system(size, rnd, balanced=True, name=str(i),
                   deadline_factor_min=0.3, sched=SchedulerType.EDF,
                   deadline_factor_max=0.7)
        for i in range(n_systems)
    ]
    utilizations = np.linspace(0.55, 0.85, n_utils)

    results = {"PD": [], "GDPA-Surr": [], "GDPA-Seq": []}
    times = {"PD": [], "GDPA-Surr": [], "GDPA-Seq": []}

    for u in utilizations:
        pd_ok = surr_ok = seq_ok = 0
        t_pd = t_surr = t_seq = 0.0

        for sys_ in systems:
            u_sys = get_system(size, rnd, balanced=True, name=f"u{u:.2f}",
                               deadline_factor_min=0.3, sched=SchedulerType.EDF,
                               deadline_factor_max=0.7, utilization=u)

            t0 = time.perf_counter()
            r = edf_pd(u_sys)
            t_pd += time.perf_counter() - t0
            pd_ok += int(r)

            t0 = time.perf_counter()
            r = edf_gdpa_surrogate(u_sys)
            t_surr += time.perf_counter() - t0
            surr_ok += int(r)

        for sys_ in systems[:3]:  # Sequential only on first 3 systems (slow)
            u_sys = get_system(size, rnd, balanced=True, name=f"u{u:.2f}-seq",
                               deadline_factor_min=0.3, sched=SchedulerType.EDF,
                               deadline_factor_max=0.7, utilization=u)

            t0 = time.perf_counter()
            r = edf_gdpa_sequential(u_sys)
            t_seq += time.perf_counter() - t0
            seq_ok += int(r)

        results["PD"].append(pd_ok / n_systems)
        results["GDPA-Surr"].append(surr_ok / n_systems)
        results["GDPA-Seq"].append(seq_ok / 3 if seq_ok > 0 else 0)
        times["PD"].append(t_pd / n_systems)
        times["GDPA-Surr"].append(t_surr / n_systems)
        times["GDPA-Seq"].append(t_seq / 3)

        print(f"u={u:.2f}  PD={pd_ok}/{n_systems}  Surr={surr_ok}/{n_systems}  Seq={seq_ok}/3")

    print("\n--- Summary ---")
    for label in ["PD", "GDPA-Surr", "GDPA-Seq"]:
        ratios = results[label]
        avg_time = np.mean(times[label])
        print(f"  {label:12s}: avg ratio={np.mean(ratios):.3f}  avg time={avg_time:.2f}s")
