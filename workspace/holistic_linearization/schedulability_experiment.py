"""
Schedulability-ratio experiment: all priority-assignment methods.

Generates 100 systems, sweeps utilisation from 0.5 to 0.9 in 20 steps,
and reports how many systems each method makes schedulable.

Methods compared:
  DM          — Deadline Monotonic (baseline)
  HOPA        — HOPA iterative heuristic
  GDPA        — Vectorised Holistic gradient descent (paper's method)
  V3-opt      — V3 surrogate gradient (p=1, one-shot)
  V1-unroll   — Phase 1: unrolled V1 gradient (N=10, tau=0.5)
  V1-anneal   — Phase 2: unrolled V1 with tau annealing
  V1-implicit — Phase 3: implicit differentiation at V1 fixed point
"""
import argparse
import os
from random import Random

import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from assignment.hopa_assignment import HOPAssignment
from examples.evaluation import SchedRatioEval
from examples.example_models import get_system
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import PriorityExtractor
from gradient_descent.stop_functions import ThresholdStopFunction, FixedIterationsStop
from gradient_descent.update_functions import NoisyAdam
from gradient_descent.interfaces import ParameterHandler
from model.linear_system import SchedulerType, LinearSystem
from vector.vector_fp import VectorFPGradientFunction, PrioritiesMatrix

from workspace.holistic_linearization.v3_gradient import V3SoftPriorityGradient
from workspace.holistic_linearization.v1_gradients import (
    V1UnrolledGradient,
    V1UnrolledAnnealedGradient,
    V1ImplicitGradient,
)
from workspace.holistic_linearization.v1_fd_gradient import V1FiniteDifferenceGradient

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


class RawPriorityHandler(ParameterHandler):
    def extract(self, system):
        return [float(t.priority) for t in system.tasks]

    def insert(self, system, x):
        for t, xi in zip(system.tasks, x):
            t.priority = max(float(xi), 0.0)


def validate(system: LinearSystem) -> bool:
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return all(f.is_schedulable() for f in system.flows)


def _make_gd_method(gradient_fn, max_iter=30, lr=0.005):
    """Factory for gradient-descent methods — returns callable."""
    def _method(system: LinearSystem) -> bool:
        analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
        param_handler = RawPriorityHandler()
        cost_function = InvslackCost(parameter_handler=param_handler,
                                      analysis=analysis)
        stop_function = FixedIterationsStop(iterations=max_iter)
        update_function = NoisyAdam(lr=lr, gamma=0.9)

        optimizer = GradientDescentOptimizer(
            parameter_handler=param_handler,
            cost_function=cost_function,
            stop_function=stop_function,
            gradient_function=gradient_fn,
            update_function=update_function,
            verbose=False,
        )

        PDAssignment(normalize=True).apply(system)
        optimizer.apply(system)
        return validate(system)
    return _method


def method_dm(system):
    PDAssignment(normalize=True).apply(system)
    return validate(system)


def method_hopa(system):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    HOPAssignment(analysis=analysis).apply(system)
    return validate(system)


def method_gdpa(system):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    param_handler = PriorityExtractor()
    cost_function = InvslackCost(parameter_handler=param_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = VectorFPGradientFunction(PrioritiesMatrix())
    update_function = NoisyAdam()
    optimizer = GradientDescentOptimizer(
        parameter_handler=param_handler, cost_function=cost_function,
        stop_function=stop_function, gradient_function=gradient_function,
        update_function=update_function, verbose=False)
    PDAssignment(normalize=True).apply(system)
    optimizer.apply(system)
    return validate(system)


def method_v3_opt(system):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    param_handler = RawPriorityHandler()
    cost_function = InvslackCost(parameter_handler=param_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=40)
    gradient_function = V3SoftPriorityGradient(tau=0.5)
    update_function = NoisyAdam(lr=0.005, gamma=0.9)
    optimizer = GradientDescentOptimizer(
        parameter_handler=param_handler, cost_function=cost_function,
        stop_function=stop_function, gradient_function=gradient_function,
        update_function=update_function, verbose=False)
    PDAssignment(normalize=True).apply(system)
    optimizer.apply(system)
    return validate(system)


def method_v1_unroll(system):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    param_handler = RawPriorityHandler()
    cost_function = InvslackCost(parameter_handler=param_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=40)
    gradient_function = V1UnrolledGradient(tau=0.5, N=10)
    update_function = NoisyAdam(lr=0.005, gamma=0.9)
    optimizer = GradientDescentOptimizer(
        parameter_handler=param_handler, cost_function=cost_function,
        stop_function=stop_function, gradient_function=gradient_function,
        update_function=update_function, verbose=False)
    PDAssignment(normalize=True).apply(system)
    optimizer.apply(system)
    return validate(system)


def method_v1_anneal(system):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    param_handler = RawPriorityHandler()
    cost_function = InvslackCost(parameter_handler=param_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=40)
    gradient_function = V1UnrolledAnnealedGradient(
        tau_0=2.0, tau_min=0.05, decay=0.92, N=10)
    update_function = NoisyAdam(lr=0.005, gamma=0.9)
    optimizer = GradientDescentOptimizer(
        parameter_handler=param_handler, cost_function=cost_function,
        stop_function=stop_function, gradient_function=gradient_function,
        update_function=update_function, verbose=False)
    PDAssignment(normalize=True).apply(system)
    optimizer.apply(system)
    return validate(system)


def method_v1_implicit(system):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    param_handler = RawPriorityHandler()
    cost_function = InvslackCost(parameter_handler=param_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=40)
    gradient_function = V1ImplicitGradient(tau=0.5, max_iters=200)
    update_function = NoisyAdam(lr=0.005, gamma=0.9)
    optimizer = GradientDescentOptimizer(
        parameter_handler=param_handler, cost_function=cost_function,
        stop_function=stop_function, gradient_function=gradient_function,
        update_function=update_function, verbose=False)
    PDAssignment(normalize=True).apply(system)
    optimizer.apply(system)
    return validate(system)


def method_v1_fd(system):
    """V1 finite-difference gradient (discrete priorities, fast surrogate)."""
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    param_handler = PriorityExtractor()
    cost_function = InvslackCost(parameter_handler=param_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=40)
    gradient_function = V1FiniteDifferenceGradient(eps=0.05)
    update_function = NoisyAdam(lr=0.005, gamma=0.9)
    optimizer = GradientDescentOptimizer(
        parameter_handler=param_handler, cost_function=cost_function,
        stop_function=stop_function, gradient_function=gradient_function,
        update_function=update_function, verbose=False)
    PDAssignment(normalize=True).apply(system)
    optimizer.apply(system)
    return validate(system)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', default=OUTPUT_DIR)
    parser.add_argument('-t', '--threads', type=int, default=4)
    args = parser.parse_args()

    rnd = Random(42)
    size = (3, 4, 3)
    n_systems = 100

    print(f"Generating {n_systems} systems ({size[0]}f x {size[1]}t x {size[2]}p) ...")
    systems = [
        get_system(size, rnd, name=str(i),
                   deadline_factor_min=0.5, deadline_factor_max=1,
                   sched=SchedulerType.FP, balanced=True)
        for i in range(n_systems)
    ]

    utilizations = np.linspace(0.5, 0.9, 20)

    methods = [
        ('DM',     method_dm),
        ('HOPA',   method_hopa),
        ('GDPA',   method_gdpa),
        ('V1-FD',  method_v1_fd),
    ]

    labels, funcs = zip(*methods)

    evaluator = SchedRatioEval(
        name='v1fd_vs_gdpa',
        labels=list(labels),
        funcs=list(funcs),
        systems=systems,
        utilizations=utilizations,
        threads=args.threads,
        output_dir=args.output_dir,
        show=False,
    )
    evaluator.run()


if __name__ == '__main__':
    main()
