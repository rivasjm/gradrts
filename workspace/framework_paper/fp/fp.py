import argparse
import os

import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis

from assignment.assignments import PDAssignment
from assignment.bf_assignment import BruteForceFPAssignment
from assignment.hopa_assignment import HOPAssignment
from examples.evaluation import SchedRatioEval
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import FPHandler
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from gradient_descent.gradient_function import SequentialGradientFunction
from model.linear_system import LinearSystem
from vector.vector_fp import VectorFPGradientFunction, PrioritiesMatrix
from workspace.framework_paper.systems import SIZES, get_systems


def gdpa_pd_fp_vector(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = FPHandler()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = VectorFPGradientFunction(PrioritiesMatrix())

    update_function = NoisyAdam()
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


def gdpa_pd_fp_seq(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = FPHandler()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = SequentialGradientFunction(cost_function=cost_function)

    update_function = NoisyAdam()
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


def pd_fp(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    pd = PDAssignment(normalize=True)
    pd.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def hopa_fp(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    hopa = HOPAssignment(analysis=analysis)
    hopa.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def bf_fp(system: LinearSystem) -> bool:
    bf = BruteForceFPAssignment(batch_size=10000)
    bf.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def build_tools(size):
    """Methods compared in the FP scenario. Brute force is only feasible for the
    smallest size, so it is included only there."""
    tools = [
        ("gdpa-vec", gdpa_pd_fp_vector),
        ("gdpa-seq", gdpa_pd_fp_seq),
        ("hopa", hopa_fp),
        ("pd", pd_fp),
    ]
    if size == 15:
        tools.append(("bf", bf_fp))
    return tools


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient FP validation")
    parser.add_argument("size", type=int, choices=sorted(SIZES), nargs="?", default=15,
                        help="System size (total number of tasks)")
    parser.add_argument("-o", "--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for generated files (default: script directory)")
    args = parser.parse_args()

    eval_name = f"fp-{args.size}"
    systems = get_systems(args.size)

    # utilizations between 50 % and 90 %
    utilizations = np.linspace(0.5, 0.9, 20)

    tools = build_tools(args.size)
    labels, funcs = zip(*tools)

    output_dir = os.path.join(args.output_dir, eval_name)
    os.makedirs(output_dir, exist_ok=True)
    runner = SchedRatioEval(eval_name, labels=labels, funcs=funcs,
                            systems=systems, utilizations=utilizations, threads=6,
                            output_dir=output_dir)
    runner.run()
