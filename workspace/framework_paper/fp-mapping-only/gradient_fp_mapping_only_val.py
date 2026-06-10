import argparse
import numpy as np
import os

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from random import Random
from functools import partial

from assignment.assignments import PDAssignment
from assignment.bf_assignment import BruteForceMappingAssignment
from examples.evaluation import SchedRatioEval
from examples.example_models import get_system
from examples.generator import unbalance_contended
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import MappingOnlyExtractor
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import VectorFPGradientFunction, MappingOnlyMatrix


def gdpa_pd_fp_mapping_only_vector(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = MappingOnlyExtractor()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100, patience=20)
    gradient_function = VectorFPGradientFunction(scenarios_builder=MappingOnlyMatrix(), sigma=3.0, cost_limit_factor=3)

    update_function = NoisyAdam(lr=1.5, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.5)
    optimizer = GradientDescentOptimizer(parameter_handler=parameter_handler,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=False)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def bf_mapping_fp(system: LinearSystem) -> bool:
    PDAssignment(normalize=True).apply(system)
    bf = BruteForceMappingAssignment(batch_size=100)
    bf.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient FP mapping-only validation")
    parser.add_argument("-o", "--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for generated files (default: script directory)")
    args = parser.parse_args()

    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    systems = [get_system(size, rnd, balanced=False, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1) for i in range(n)]

    utilizations = np.linspace(0.5, 0.9, 20)

    tools = [
        ("gdpa-mapping", gdpa_pd_fp_mapping_only_vector)
        # ("bf-mapping", bf_mapping_fp)
    ]

    labels, funcs = zip(*tools)
    runner = SchedRatioEval("gradient_fp_mapping_only_validation", labels=labels, funcs=funcs,
                            preprocessor=unbalance_contended,
                            systems=systems, utilizations=utilizations, threads=6,
                            output_dir=args.output_dir, show=False)
    runner.run()
