import unittest
from random import Random

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.hopa_assignment import HOPAssignment
from assignment.assignments import PDAssignment
from examples.example_models import get_palencia_system
from examples.generator import generate_system
from model.linear_system import SchedulerType


class HolisticTest(unittest.TestCase):
    def test_palencia(self):
        system = get_palencia_system()
        flow1 = system['flow1']
        flow2 = system['flow2']
        cpu1 = next((p for p in system.processors if p.name == "cpu1"), None)
        cpu2 = next((p for p in system.processors if p.name == "cpu2"), None)
        network = next((p for p in system.processors if p.name == "network"), None)

        # analyze
        system.apply(HolisticFPAnalysis())

        #
        self.assertEqual(flow1.wcrt, 42)
        self.assertEqual(flow2.wcrt, 30)
        self.assertTrue(system.is_schedulable())
        self.assertAlmostEqual(cpu1.utilization, 0.416, delta=0.001)
        self.assertAlmostEqual(cpu2.utilization, 0.791, delta=0.001)
        self.assertAlmostEqual(network.utilization, 0.316, delta=0.001)
        self.assertAlmostEqual(system.max_utilization, 0.791, delta=0.001)
        self.assertAlmostEqual(system.utilization, 0.508, delta=0.001)

        #
        # print(system.processors)

    def test_random(self):
        random = Random(10)
        pd = PDAssignment()
        holistic = HolisticFPAnalysis()
        hopa = HOPAssignment(analysis=HolisticFPAnalysis(reset=False), over_iterations=10, verbose=True)

        utilization = 0.85
        system = generate_system(random,
                                 n_flows=random.randint(1, 10),
                                 n_tasks=random.randint(1, 10),
                                 n_procs=random.randint(1, 5),
                                 sched=SchedulerType.FP,
                                 utilization=utilization,
                                 period_min=100, period_max=100 * random.uniform(2.0, 1000.0),
                                 deadline_factor_min=0.5, deadline_factor_max=2)

        # this one should be schedulable at the third iteration
        # in the first iteration it triggers the stop factor
        system.apply(hopa)
        system.apply(holistic)
        print(system.is_schedulable())
        print(system.slack)
        self.assertAlmostEqual(system.slack, 0.28803, delta=0.00001)  # should be from iteration 8 of 12


if __name__ == '__main__':
    unittest.main()