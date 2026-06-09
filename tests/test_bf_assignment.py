import itertools
import math
import unittest

import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from assignment.bf_assignment import BruteForceAssignment
from examples.example_models import get_palencia_system, get_three_tasks, get_system
from examples.generator import set_utilization
from random import Random


class BruteForceTest(unittest.TestCase):

    # -- helpers --

    @staticmethod
    def _make_schedulable(system, prios):
        for prio, task in zip(prios, system.tasks):
            task.priority = prio

    # -- space size & mapping --

    def test_space_size_single_proc(self):
        system = get_three_tasks()
        bf = BruteForceAssignment()
        self.assertEqual(bf._space_size(system), math.factorial(3))

    def test_space_size_palencia(self):
        system = get_palencia_system()
        bf = BruteForceAssignment()
        self.assertEqual(bf._space_size(system), 8)

    def test_task_mapping_is_bijection(self):
        system = get_palencia_system()
        bf = BruteForceAssignment()
        proc_tasks_list = [proc.tasks for proc in system.processors]
        mapping = bf._build_task_mapping(system.tasks, proc_tasks_list)
        self.assertEqual(len(mapping), len(system.tasks))
        self.assertEqual(len(set(mapping)), len(system.tasks))

    # -- priority matrix correctness --

    def test_priority_matrix_single_scenario(self):
        system = get_palencia_system()
        n = len(system.tasks)
        bf = BruteForceAssignment()
        scenarios = np.array([range(1, n + 1)]).T
        pm = bf._build_priority_matrix(system, scenarios)
        self.assertEqual(pm.shape, (1, n, n))
        self.assertFalse(np.any(np.diagonal(pm[0])))

    def test_priority_matrix_same_proc_only(self):
        system = get_palencia_system()
        bf = BruteForceAssignment()
        procs = system.processors
        tasks = system.tasks
        scenarios = np.array([range(1, len(tasks) + 1)]).T
        pm = bf._build_priority_matrix(system, scenarios)[0]
        for i, ti in enumerate(tasks):
            for j, tj in enumerate(tasks):
                if ti.processor != tj.processor:
                    self.assertFalse(pm[i, j],
                                     f"Cross-processor interference at ({i},{j})")

    # -- known system: Palencia --

    def test_palencia_schedulable(self):
        system = get_palencia_system()
        bf = BruteForceAssignment(batch_size=100)
        bf.apply(system)
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)
        self.assertTrue(system.is_schedulable())
        self.assertTrue(bf.schedulable)

    def test_palencia_solution_matches_sequential(self):
        system = get_palencia_system()
        tasks = system.tasks

        bf = BruteForceAssignment(batch_size=100)
        bf.apply(system)
        prios = [t.priority for t in tasks]

        system2 = get_palencia_system()
        PDAssignment(normalize=True).apply(system2)
        self._make_schedulable(system2, prios)
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system2)

        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)

        for i, t1, t2 in zip(range(len(tasks)), system.tasks, system2.tasks):
            self.assertAlmostEqual(t1.wcrt, t2.wcrt, delta=0.001,
                                   msg=f"Task {i} WCRT mismatch: {t1.wcrt:.4f} vs {t2.wcrt:.4f}")

    # -- exhaustive search property --

    def test_single_proc_deterministic(self):
        system = get_three_tasks()
        bf = BruteForceAssignment()
        self.assertEqual(bf._space_size(system), 6)
        bf.apply(system)
        self.assertTrue(bf.schedulable)
        self.assertGreater(bf.exec_time.exec_time, 0)

    def test_pd_is_subset(self):
        for seed in range(10):
            with self.subTest(seed=seed):
                rnd = Random(seed)
                system = get_system((2, 2, 2), random=rnd, utilization=0.5, balanced=True)
                analysis = HolisticFPAnalysis(limit_factor=10, reset=False)

                pd = PDAssignment(normalize=True)
                pd.apply(system)
                analysis.apply(system)
                if system.is_schedulable():
                    bf = BruteForceAssignment(batch_size=100)
                    bf.apply(system)
                    analysis.apply(system)
                    self.assertTrue(system.is_schedulable(),
                                    f"BF missed solution PD found (seed={seed})")

    def test_exhaustion_marks_not_schedulable(self):
        rnd = Random(99)
        system = get_system((2, 2, 2), random=rnd, utilization=0.95, balanced=True)
        bf = BruteForceAssignment(batch_size=100)
        bf.apply(system)
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)
        self.assertFalse(bf.schedulable)

    # -- attributes --

    def test_exec_time_populated(self):
        system = get_palencia_system()
        bf = BruteForceAssignment()
        bf.apply(system)
        self.assertTrue(bf.exec_time.has_time())

    def test_iterations_to_sched(self):
        system = get_palencia_system()
        bf = BruteForceAssignment(batch_size=1)
        bf.apply(system)
        self.assertGreater(bf.iterations_to_sched, 0)

    def test_space_size_attribute(self):
        system = get_palencia_system()
        bf = BruteForceAssignment()
        bf.apply(system)
        self.assertEqual(bf.space_size, 8)

    # -- batch processing splits correctly --

    def test_batch_smaller_than_space(self):
        system = get_palencia_system()
        bf = BruteForceAssignment(batch_size=3)
        bf.apply(system)
        self.assertTrue(bf.schedulable)
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)
        self.assertTrue(system.is_schedulable())


if __name__ == '__main__':
    unittest.main()
