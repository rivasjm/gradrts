import argparse
import os
from functools import partial

import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis

from assignment.assignments import PDAssignment
from assignment.hopa_assignment import HOPAssignment
from examples.evaluation import SchedRatioEval
from examples.generator import set_system_utilization, set_utilization
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import FPMappingHandler
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import VectorFPGradientFunction, MappingPrioritiesMatrix
from workspace.framework_paper.systems import SIZES, get_systems


def hopa_fp(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    HOPAssignment(analysis=analysis).apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def pd_fp(system: LinearSystem) -> bool:
    pd = PDAssignment(normalize=True)
    pd.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def gdpa_mapping_fp(system: LinearSystem, limit: int) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = FPMappingHandler()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=limit)
    gradient_function = VectorFPGradientFunction(scenarios_builder=MappingPrioritiesMatrix())

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Gradient FP+mapping validation")
    parser.add_argument("size", type=int, choices=sorted(SIZES), nargs="?", default=15,
                        help="System size (total number of tasks)")
    parser.add_argument("--unbalanced", action="store_true",
                        help="Start from an unbalanced initial mapping with uneven per-processor "
                             "load, and sweep utilization while keeping that mapping")
    parser.add_argument("-o", "--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for generated files (default: script directory)")
    args = parser.parse_args()

    eval_name = f"map-{args.size}" if not args.unbalanced else f"map-unbalanced-{args.size}"
    systems = get_systems(args.size, balanced=not args.unbalanced)
    utilization_func = set_system_utilization if args.unbalanced else set_utilization

    # utilizations between 50 % and 90 %
    utilizations = np.linspace(0.5, 0.9, 20)

    tools = [
        ("pd", pd_fp),
        ("hopa", hopa_fp),
        ("gdpa-50", partial(gdpa_mapping_fp, limit=50)),
        ("gdpa-100", partial(gdpa_mapping_fp, limit=100)),
        ("gdpa-200", partial(gdpa_mapping_fp, limit=200)),
    ]

    labels, funcs = zip(*tools)
    output_dir = os.path.join(args.output_dir, eval_name)
    os.makedirs(output_dir, exist_ok=True)
    runner = SchedRatioEval(eval_name, labels=labels, funcs=funcs,
                            systems=systems, utilizations=utilizations, threads=6,
                            utilization_func=utilization_func,
                            output_dir=output_dir)
    runner.run()
