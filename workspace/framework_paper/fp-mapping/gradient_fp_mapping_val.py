import argparse
import numpy as np
import os
from functools import partial

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from random import Random

from assignment.assignments import PDAssignment
from assignment.hopa_assignment import HOPAssignment
from examples.evaluation import SchedRatioEval
from examples.example_models import get_system
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import MappingPriorityExtractor
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import VectorFPGradientFunction, MappingPrioritiesMatrix


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
    parameter_handler = MappingPriorityExtractor()
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
    parser.add_argument("-o", "--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory for generated files (default: script directory)")
    args = parser.parse_args()

    # create population of examples
    rnd = Random(42)
    size = (5, 3, 3)  # flows, tasks, procs
    n = 50
    systems = [get_system(size, rnd, balanced=False, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1,
                          period_min=100, period_max=1000) for i in range(n)]

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
    runner = SchedRatioEval("gradient_fp_mapping_eval", labels=labels, funcs=funcs,
                            # preprocessor=unbalance_contended,
                            systems=systems, utilizations=utilizations, threads=6,
                            output_dir=args.output_dir)
    runner.run()
