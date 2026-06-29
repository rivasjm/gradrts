"""Schedulability experiment comparing DM, HOPA, GDPA and V1-TopK.

Mirrors ``schedulability_experiment.py`` but adds the V1-TopK method
(non-iterative random exploration ranked by vectorised V1 + Holistic
validation of the top-K).
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
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import SchedulerType, LinearSystem
from vector.vector_fp import VectorFPGradientFunction, PrioritiesMatrix

from workspace.holistic_linearization.v1_topk import make_v1_topk_method

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def validate(system: LinearSystem) -> bool:
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return all(f.is_schedulable() for f in system.flows)


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

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


class V1TopKMethod:
    """Picklable callable wrapping ``v1_topk_assign`` for SchedRatioEval."""

    def __init__(self, n_candidates=100, k=5, limit_factor=10, perturbations=0):
        self.n_candidates = n_candidates
        self.k = k
        self.limit_factor = limit_factor
        self.perturbations = perturbations

    def __call__(self, system: LinearSystem) -> bool:
        from workspace.holistic_linearization.v1_topk import v1_topk_assign
        return v1_topk_assign(
            system,
            n_candidates=self.n_candidates,
            k=self.k,
            limit_factor=self.limit_factor,
            perturbations=self.perturbations,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', default=OUTPUT_DIR)
    parser.add_argument('-t', '--threads', type=int, default=4)
    parser.add_argument('-n', '--n-systems', type=int, default=100)
    parser.add_argument('--nc', type=int, default=100, help='V1-TopK n_candidates')
    parser.add_argument('--k', type=int, default=5,    help='V1-TopK K survivors')
    parser.add_argument('--pert', type=int, default=0,
                        help='V1-TopK perturbations per candidate '
                             '(0 = uniform random; >0 = DM-anchored swaps)')
    args = parser.parse_args()

    rnd = Random(42)
    size = (3, 4, 3)
    n_systems = args.n_systems

    print(f"Generating {n_systems} systems ({size[0]}f x {size[1]}t x {size[2]}p) ...")
    systems = [
        get_system(size, rnd, name=str(i),
                   deadline_factor_min=0.5, deadline_factor_max=1,
                   sched=SchedulerType.FP, balanced=True)
        for i in range(n_systems)
    ]

    utilizations = np.linspace(0.5, 0.9, 20)

    methods = [
        ('DM',         method_dm),
        ('HOPA',       method_hopa),
        ('GDPA',       method_gdpa),
        ('V1-TopK',    V1TopKMethod(n_candidates=args.nc, k=args.k,
                            perturbations=args.pert)),
    ]

    labels, funcs = zip(*methods)

    evaluator = SchedRatioEval(
        name='v1topk_vs_baseline',
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