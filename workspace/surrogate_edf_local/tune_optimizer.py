"""Test optimizer configurations to improve surrogate schedulability."""

import copy
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


def optimise_and_check(system, grad_fn, max_iters, patience):
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
    stop_fn = ThresholdStopFunction(limit=max_iters, patience=patience)
    update_fn = NoisyAdam()
    optimizer = GradientDescentOptimizer(
        parameter_handler=ph, cost_function=cost_fn,
        stop_function=stop_fn, gradient_function=grad_fn,
        update_function=update_fn, verbose=False,
    )
    PDAssignment().apply(system)
    t0 = time.perf_counter()
    optimizer.apply(system)
    elapsed = time.perf_counter() - t0
    HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
    final_cost = stop_fn.solution_cost()
    schedulable = system.is_schedulable()
    return schedulable, final_cost, elapsed


def main():
    rnd = Random(42)
    n_sys = 10
    systems = [
        get_system((3, 4, 3), rnd, balanced=True, name=str(i),
                   deadline_factor_min=0.3, sched=SchedulerType.EDF,
                   deadline_factor_max=0.7)
        for i in range(n_sys)
    ]

    # Test configurations
    configs = [
        # (name, max_iters, patience, tau, temp_max)
        ("base (50it,tau=0.1)",      50,  15, 0.1, 0.1),
        ("100it,tau=0.1",           100,  30, 0.1, 0.1),
        ("200it,tau=0.1",           200,  50, 0.1, 0.1),
        ("200it,tau=0.1,temp=0.01", 200,  50, 0.1, 0.01),
        ("200it,tau=0.05",          200,  50, 0.05, 0.1),
        ("100it,tau=0.1,temp=0.01", 100,  30, 0.1, 0.01),
    ]

    print(f"{'Config':<28s} {'Sched':>6s} {'Cost':>10s} {'Time':>8s}")
    print("-" * 55)

    for cfg_name, max_it, pat, tau, tmax in configs:
        sched_count = 0
        costs = []
        times = []
        for sys_template in systems:
            s = copy.deepcopy(sys_template)
            grad_fn = SurrogateEDFGradient(
                tau=tau, N_w=10, N_jitter=2, M_psi=50,
                temperature_max=tmax, grad_clip=1.0,
            )
            ok, cost, elapsed = optimise_and_check(s, grad_fn, max_it, pat)
            sched_count += int(ok)
            costs.append(cost)
            times.append(elapsed)

        print(f"{cfg_name:<28s} {sched_count:4d}/{n_sys} {np.mean(costs):+10.4f} {np.mean(times):7.2f}s")

    # Also run sequential for reference on same systems
    print("-" * 55)
    sched_seq = 0
    costs_seq = []
    times_seq = []
    for sys_template in systems[:5]:  # only 5 for time
        s = copy.deepcopy(sys_template)
        analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
        ph = DeadlineExtractor()
        cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
        grad_fn = SequentialGradientFunction(cost_function=cost_fn, sigma=1.5)
        ok, cost, elapsed = optimise_and_check(s, grad_fn, 50, 15)
        sched_seq += int(ok)
        costs_seq.append(cost)
        times_seq.append(elapsed)
    print(f"{'GDPA-Seq (50it)':<28s} {sched_seq:4d}/{min(5,n_sys)} {np.mean(costs_seq):+10.4f} {np.mean(times_seq):7.2f}s")


if __name__ == "__main__":
    main()
