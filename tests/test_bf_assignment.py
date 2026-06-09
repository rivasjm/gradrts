import itertools
import math
import unittest

import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from assignment.bf_assignment import BruteForceFPAssignment, BruteForceFPMappingAssignment, BruteForceMappingAssignment
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
        bf = BruteForceFPAssignment()
        self.assertEqual(bf._space_size(system), math.factorial(3))

    def test_space_size_palencia(self):
        system = get_palencia_system()
        bf = BruteForceFPAssignment()
        self.assertEqual(bf._space_size(system), 8)

    def test_task_mapping_is_bijection(self):
        system = get_palencia_system()
        bf = BruteForceFPAssignment()
        proc_tasks_list = [proc.tasks for proc in system.processors]
        mapping = bf._build_task_mapping(system.tasks, proc_tasks_list)
        self.assertEqual(len(mapping), len(system.tasks))
        self.assertEqual(len(set(mapping)), len(system.tasks))

    # -- priority matrix correctness --

    def test_priority_matrix_single_scenario(self):
        system = get_palencia_system()
        n = len(system.tasks)
        bf = BruteForceFPAssignment()
        scenarios = np.array([range(1, n + 1)]).T
        pm = bf._build_priority_matrix(system, scenarios)
        self.assertEqual(pm.shape, (1, n, n))
        self.assertFalse(np.any(np.diagonal(pm[0])))

    def test_priority_matrix_same_proc_only(self):
        system = get_palencia_system()
        bf = BruteForceFPAssignment()
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
        bf = BruteForceFPAssignment(batch_size=100)
        bf.apply(system)
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)
        self.assertTrue(system.is_schedulable())
        self.assertTrue(bf.schedulable)

    def test_palencia_solution_matches_sequential(self):
        system = get_palencia_system()
        tasks = system.tasks

        bf = BruteForceFPAssignment(batch_size=100)
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
        bf = BruteForceFPAssignment()
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
                    bf = BruteForceFPAssignment(batch_size=100)
                    bf.apply(system)
                    analysis.apply(system)
                    self.assertTrue(system.is_schedulable(),
                                    f"BF missed solution PD found (seed={seed})")

    def test_exhaustion_marks_not_schedulable(self):
        rnd = Random(99)
        system = get_system((2, 2, 2), random=rnd, utilization=0.95, balanced=True)
        bf = BruteForceFPAssignment(batch_size=100)
        bf.apply(system)
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)
        self.assertFalse(bf.schedulable)

    # -- attributes --

    def test_exec_time_populated(self):
        system = get_palencia_system()
        bf = BruteForceFPAssignment()
        bf.apply(system)
        self.assertTrue(bf.exec_time.has_time())

    def test_iterations_to_sched(self):
        system = get_palencia_system()
        bf = BruteForceFPAssignment(batch_size=1)
        bf.apply(system)
        self.assertGreater(bf.iterations_to_sched, 0)

    def test_space_size_attribute(self):
        system = get_palencia_system()
        bf = BruteForceFPAssignment()
        bf.apply(system)
        self.assertEqual(bf.space_size, 8)

    # -- batch processing splits correctly --

    def test_batch_smaller_than_space(self):
        system = get_palencia_system()
        bf = BruteForceFPAssignment(batch_size=3)
        bf.apply(system)
        self.assertTrue(bf.schedulable)
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)
        self.assertTrue(system.is_schedulable())


class BruteForceMappingTest(unittest.TestCase):

    def test_space_size_two_procs_two_tasks(self):
        system = get_system((1, 2, 2), random=Random(42), utilization=0.5, balanced=True)
        bf = BruteForceFPMappingAssignment()
        bf.apply(system)
        self.assertEqual(bf.space_size, 6)

    def test_palencia_schedulable(self):
        system = get_palencia_system()
        bf = BruteForceFPMappingAssignment(batch_size=100)
        bf.apply(system)
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)
        self.assertTrue(system.is_schedulable())
        self.assertTrue(bf.schedulable)

    def test_palencia_matches_sequential(self):
        system = get_palencia_system()
        tasks = system.tasks

        bf = BruteForceFPMappingAssignment(batch_size=100)
        bf.apply(system)
        bf_mapping = [system.processors.index(t.processor) for t in tasks]
        bf_prios = [t.priority for t in tasks]

        system2 = get_palencia_system()
        PDAssignment(normalize=True).apply(system2)
        for task, proc_idx, prio in zip(system2.tasks, bf_mapping, bf_prios):
            task.processor = system2.processors[proc_idx]
            task.priority = prio
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system2)

        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)

        for i, t1, t2 in zip(range(len(tasks)), system.tasks, system2.tasks):
            self.assertAlmostEqual(t1.wcrt, t2.wcrt, delta=0.001,
                                   msg=f"Task {i} WCRT mismatch: {t1.wcrt:.4f} vs {t2.wcrt:.4f}")

    def test_pd_is_subset(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                rnd = Random(seed)
                system = get_system((1, 2, 2), random=rnd, utilization=0.5, balanced=True)
                analysis = HolisticFPAnalysis(limit_factor=10, reset=False)

                pd = PDAssignment(normalize=True)
                pd.apply(system)
                analysis.apply(system)
                if system.is_schedulable():
                    bf = BruteForceFPMappingAssignment(batch_size=100)
                    bf.apply(system)
                    analysis.apply(system)
                    self.assertTrue(system.is_schedulable(),
                                    f"BF missed solution PD found (seed={seed})")

    def test_priority_matrix_respects_mapping(self):
        system = get_system((1, 3, 2), random=Random(42), utilization=0.5, balanced=True)
        n = len(system.tasks)
        mapping = (0, 1, 1)
        priorities = (3, 2, 1)
        pm = BruteForceFPMappingAssignment._single_priority_matrix(mapping, priorities, n)
        self.assertAlmostEqual(pm[0, 1], 0.0)
        self.assertAlmostEqual(pm[0, 2], 0.0)
        self.assertGreater(pm[2, 1], 0.0)

    def test_exec_time_and_attributes(self):
        system = get_palencia_system()
        bf = BruteForceFPMappingAssignment(batch_size=100)
        bf.apply(system)
        self.assertTrue(bf.exec_time.has_time())
        self.assertGreater(bf.space_size, 0)
        self.assertTrue(bf.schedulable)
        self.assertGreater(bf.iterations_to_sched, 0)


class BruteForceMappingOnlyTest(unittest.TestCase):

    def test_space_size(self):
        system = get_system((1, 2, 2), random=Random(42), utilization=0.5, balanced=True)
        bf = BruteForceMappingAssignment()
        bf.apply(system)
        self.assertEqual(bf.space_size, 4)

    def test_palencia_schedulable(self):
        system = get_palencia_system()
        PDAssignment(normalize=True).apply(system)
        bf = BruteForceMappingAssignment(batch_size=100)
        bf.apply(system)
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)
        self.assertTrue(system.is_schedulable())
        self.assertTrue(bf.schedulable)

    def test_palencia_matches_sequential(self):
        system = get_palencia_system()
        PDAssignment(normalize=True).apply(system)
        tasks = system.tasks

        bf = BruteForceMappingAssignment(batch_size=100)
        bf.apply(system)
        bf_mapping = [system.processors.index(t.processor) for t in tasks]
        bf_prios = [t.priority for t in tasks]

        system2 = get_palencia_system()
        PDAssignment(normalize=True).apply(system2)
        for task, proc_idx, prio in zip(system2.tasks, bf_mapping, bf_prios):
            task.processor = system2.processors[proc_idx]
            task.priority = prio
        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system2)

        HolisticFPAnalysis(limit_factor=10, reset=True).apply(system)

        for i, t1, t2 in zip(range(len(tasks)), system.tasks, system2.tasks):
            self.assertAlmostEqual(t1.wcrt, t2.wcrt, delta=0.001,
                                   msg=f"Task {i} WCRT mismatch: {t1.wcrt:.4f} vs {t2.wcrt:.4f}")

    def test_pd_is_subset(self):
        for seed in range(5):
            with self.subTest(seed=seed):
                rnd = Random(seed)
                system = get_system((1, 2, 2), random=rnd, utilization=0.5, balanced=True)
                analysis = HolisticFPAnalysis(limit_factor=10, reset=False)

                pd = PDAssignment(normalize=True)
                pd.apply(system)
                analysis.apply(system)
                if system.is_schedulable():
                    bf = BruteForceMappingAssignment(batch_size=100)
                    bf.apply(system)
                    analysis.apply(system)
                    self.assertTrue(system.is_schedulable(),
                                    f"BF missed solution PD found (seed={seed})")

    def test_priorities_unchanged(self):
        system = get_system((1, 3, 2), random=Random(42), utilization=0.5, balanced=True)
        PDAssignment(normalize=True).apply(system)
        original_priorities = [t.priority for t in system.tasks]
        bf = BruteForceMappingAssignment(batch_size=100)
        bf.apply(system)
        for task, orig_prio in zip(system.tasks, original_priorities):
            self.assertAlmostEqual(task.priority, orig_prio, delta=0.001,
                msg=f"Task {task.name} priority changed: {orig_prio} -> {task.priority}")

    def test_exec_time_and_attributes(self):
        system = get_palencia_system()
        bf = BruteForceMappingAssignment(batch_size=100)
        bf.apply(system)
        self.assertTrue(bf.exec_time.has_time())
        self.assertGreater(bf.space_size, 0)
        self.assertTrue(bf.schedulable)
        self.assertGreater(bf.iterations_to_sched, 0)


if __name__ == '__main__':
    unittest.main()
