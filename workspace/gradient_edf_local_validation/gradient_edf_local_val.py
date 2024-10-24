from random import Random

import numpy as np

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment, EQFAssignment
from examples.evaluation import SchedRatioEval
from examples.example_models import get_system
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import AvgSeparationDelta, SequentialGradientFunction
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import DeadlineExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem, SchedulerType


def item(system, assignment, test):
    assignment.apply(system)
    test.apply(system)
    return system.is_schedulable()


def edf_local_pd(system: LinearSystem) -> bool:
    return item(system, PDAssignment(), HolisticLocalEDFAnalysis(limit_factor=10, reset=False))


def edf_local_eqf(system: LinearSystem) -> bool:
    return item(system,  EQFAssignment(), HolisticLocalEDFAnalysis(limit_factor=10, reset=False))


def edf_local_gdpa(system: LinearSystem) -> bool:
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    parameter_handler = DeadlineExtractor()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = SequentialGradientFunction(cost_function=cost_function, sigma=1.5)
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
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    systems = [get_system(size, rnd, balanced=True, name=str(i),
                          deadline_factor_min=0.5, sched=SchedulerType.EDF,
                          deadline_factor_max=1) for i in range(n)]

    # utilizations between 50 % and 90 %
    utilizations = np.linspace(0.5, 0.9, 20)

    tools = [("EDF-L PD", edf_local_pd),
             ("EDF-L EQF", edf_local_eqf),
             ("EDF-L GDPA", edf_local_gdpa)]

    labels, funcs = zip(*tools)
    runner = SchedRatioEval("gradient_edf_local_validation", labels=labels, funcs=funcs,
                            systems=systems, utilizations=utilizations, threads=6)
    runner.run()
