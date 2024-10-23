import unittest

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from examples.examples_special import get_validation_example
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import PriorityExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import VectorFPGradientFunction, PrioritiesMatrix


class GradientTest(unittest.TestCase):

    def test_validation(self):
        system = get_validation_example()
        print(system)
        gdpa_pd_fp_vector(system)
        self.assertFalse(system.is_schedulable())


if __name__ == '__main__':
    unittest.main()


def gdpa_pd_fp_vector(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = PriorityExtractor()
    cost_function = InvslackCost(param_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100)
    gradient_function = VectorFPGradientFunction(PrioritiesMatrix())

    update_function = NoisyAdam()
    optimizer = GradientDescentOptimizer(parameter_handler=parameter_handler,
                                        cost_function=cost_function,
                                        stop_function=stop_function,
                                        gradient_function=gradient_function,
                                        update_function=update_function,
                                        verbose=True)

    pd = PDAssignment(normalize=True)
    pd.apply(system)
    optimizer.apply(system)
    analysis.apply(system)
    return system.is_schedulable()