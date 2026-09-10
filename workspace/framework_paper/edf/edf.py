import argparse
import os

import numpy as np

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from assignment.hopa_assignment import HOPAssignment
from examples.evaluation import SchedRatioEval
from examples.generator import to_edf
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import SequentialGradientFunction
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import DeadlineHandler
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from workspace.framework_paper.systems import SIZES, get_systems


def item(system, assignment, test):
    assignment.apply(system)
    test.apply(system)
    return system.is_schedulable()


def edf_local_pd(system: LinearSystem) -> bool:
    return item(system, PDAssignment(), HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


def edf_local_hopa(system: LinearSystem) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    return item(system, HOPAssignment(analysis=analysis), HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


def edf_local_gdpa(system: LinearSystem) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False, max_time=None)
    parameter_handler = DeadlineHandler()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100, max_time=120)
    gradient_function = SequentialGradientFunction(cost_function=cost_function)
    update_function = NoisyAdam()
    optimizer = GradientDescentOptimizer(parameter_handler=parameter_handler,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=False)

    PDAssignment().apply(system)
    return item(system, optimizer, HolisticLocalEDFAnalysis(limit_factor=1, reset=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient EDF local validation")
    parser.add_argument("size", type=int, choices=sorted(SIZES), nargs="?", default=15,
                        help="System size (total number of tasks)")
    parser.add_argument("-o", "--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for generated files (default: script directory)")
    args = parser.parse_args()

    eval_name = f"edf-{args.size}"
    systems = get_systems(args.size)
    for system in systems:
        to_edf(system)

    # utilizations between 50 % and 90 %
    utilizations = np.linspace(0.5, 0.9, 20)

    tools = [("pd", edf_local_pd),
             ("hopa", edf_local_hopa),
             ("gdpa", edf_local_gdpa)]

    labels, funcs = zip(*tools)
    output_dir = os.path.join(args.output_dir, eval_name)
    os.makedirs(output_dir, exist_ok=True)
    runner = SchedRatioEval(eval_name, labels=labels, funcs=funcs,
                            systems=systems, utilizations=utilizations, threads=6,
                            output_dir=output_dir)
    runner.run()
