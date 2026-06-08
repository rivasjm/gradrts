import unittest

import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from examples.examples_special import get_validation_example
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import gradient_inputs_from_deltas
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import PriorityExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import VectorFPGradientFunction, PrioritiesMatrix, VectorHolisticFPAnalysis, ResultsCache


class GradientTest(unittest.TestCase):

    def test_validation(self):
        system = get_validation_example()
        print(system)
        gdpa_pd_fp_vector(system)
        self.assertFalse(system.is_schedulable())

    def test_vectorized_matches_sequential(self):
        system_seq = get_validation_example()
        pd = PDAssignment(normalize=True)
        pd.apply(system_seq)
        seq = HolisticFPAnalysis(limit_factor=10, reset=False)
        seq.apply(system_seq)
        seq_wcrts = [t.wcrt for t in system_seq.tasks]

        system_vec = get_validation_example()
        pd.apply(system_vec)
        vec = VectorHolisticFPAnalysis(limit_factor=10, cache=ResultsCache())
        vec.apply(system_vec)
        vec_wcrts = [t.wcrt for t in system_vec.tasks]

        for i, (s, v) in enumerate(zip(seq_wcrts, vec_wcrts)):
            self.assertAlmostEqual(s, v, delta=0.001,
                                   msg=f"Task {i} WCRT mismatch: seq={s:.6f} vec={v:.6f}")

    def test_priorities_matrix_scenarios(self):
        system = get_validation_example()
        n_tasks = len(system.tasks)
        x = [1.0 + 0.05 * i for i in range(n_tasks)]  # close-spaced to make delta cross boundaries
        delta = 0.1
        inputs = gradient_inputs_from_deltas(x, [delta] * len(x))

        pm = PrioritiesMatrix().apply(system, inputs)
        self.assertEqual(pm.shape, (len(inputs), n_tasks, n_tasks))

        differing = 0
        for i in range(len(x)):
            if np.any(pm[2 * i] != pm[2 * i + 1]):
                differing += 1
        self.assertGreater(differing, 0,
                           "No perturbed scenario pair produced different priority matrices")


if __name__ == '__main__':
    unittest.main()


def gdpa_pd_fp_vector(system: LinearSystem) -> bool:
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = PriorityExtractor()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
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