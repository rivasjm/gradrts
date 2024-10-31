import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from random import Random
from functools import partial

from assignment.assignments import PDAssignment, EQSAssignment, EQFAssignment
from assignment.hopa_assignment import HOPAssignment
from examples import generator
from examples.evaluation import SchedRatioEval
from examples.example_models import get_system
from examples.generator import unbalance, set_utilization
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import PriorityExtractor, MappingPriorityExtractor
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import VectorFPGradientFunction, PrioritiesMatrix


def gdpa_fp_mapping_vector(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = MappingPriorityExtractor()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = VectorFPGradientFunction(sigma=1.5)

    update_function = NoisyAdam(lr=1.5, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.5)
    optimizer = GradientDescentOptimizer(parameter_handler=parameter_handler,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=True)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def get_test_system():
    # create population of examples
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 1
    systems = [get_system(size, rnd, balanced=False, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1) for i in range(n)]

    # utilizations between 50 % and 90 %
    utilizations = np.linspace(0.5, 0.9, 20)
    system = systems[0]
    set_utilization(system, 0.5)
    unbalance(system)
    return system


if __name__ == '__main__':
    system = get_test_system()
    print(system)
    sched = gdpa_fp_mapping_vector(system)
    print(sched)