import argparse
import os
from functools import partial

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


def edf_local_gdpa(system: LinearSystem, max_time: float = None) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False, max_time=max_time)
    parameter_handler = DeadlineHandler()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100, max_time=max_time)
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


def run_size(size: int, max_time: float, base_output_dir: str) -> None:
    eval_name = f"edf-{size}"
    systems = get_systems(size)
    for system in systems:
        to_edf(system)

    # utilizations between 50 % and 90 %
    utilizations = np.linspace(0.5, 0.9, 20)

    tools = [("pd", edf_local_pd),
             ("hopa", edf_local_hopa),
             ("gdpa", partial(edf_local_gdpa, max_time=max_time))]

    labels, funcs = zip(*tools)
    output_dir = os.path.join(base_output_dir, eval_name)
    os.makedirs(output_dir, exist_ok=True)
    runner = SchedRatioEval(eval_name, labels=labels, funcs=funcs,
                            systems=systems, utilizations=utilizations, threads=6,
                            output_dir=output_dir)
    runner.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient EDF local validation")
    parser.add_argument("size", type=int, choices=sorted(SIZES), nargs="*", default=None,
                        help="System size(s) (total number of tasks); can be repeated. "
                             "Default: every size")
    parser.add_argument("--max-time", type=float, default=None, metavar="SECONDS",
                        help="Wall-clock budget per gdpa run, applied to both the analysis and "
                             "the optimizer stop function; the optimizer returns its best "
                             "solution so far when exceeded. Default: no limit.")
    parser.add_argument("-o", "--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for generated files (default: script directory)")
    args = parser.parse_args()

    for size in (args.size or sorted(SIZES)):
        run_size(size, args.max_time, args.output_dir)
