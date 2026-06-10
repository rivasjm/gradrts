import unittest

from examples.example_models import get_system
from examples.generator import unbalance_contended, set_utilization
from random import Random


class UnbalanceContendedTest(unittest.TestCase):

    def setUp(self):
        self.rnd = Random(42)
        self.size = (3, 4, 3)  # 3 flows, 4 tasks, 3 processors

    def _make_system(self, utilization=0.5):
        s = get_system(self.size, self.rnd, balanced=False, utilization=0.5)
        set_utilization(s, utilization)
        return s

    def _processor_loads(self, system):
        procs = system.processors
        loads = [0.0] * len(procs)
        counts = [0] * len(procs)
        for t in system.tasks:
            pi = procs.index(t.processor)
            loads[pi] += t.utilization
            counts[pi] += 1
        return loads, counts

    # -- basic properties --

    def test_no_processor_exceeds_max_utilization(self):
        for u in [0.5, 0.6, 0.7, 0.8, 0.9]:
            with self.subTest(utilization=u):
                s = self._make_system(u)
                unbalance_contended(s, max_utilization=0.95)
                loads, _ = self._processor_loads(s)
                for load in loads:
                    self.assertLess(load, 0.95,
                                    f"Processor load {load:.3f} >= 0.95 at u={u}")

    def test_all_tasks_assigned(self):
        s = self._make_system(0.7)
        unbalance_contended(s)
        for t in s.tasks:
            self.assertIsNotNone(t.processor, f"Task {t.name} has no processor")

    def test_total_utilization_preserved(self):
        for u in [0.5, 0.7, 0.9]:
            with self.subTest(utilization=u):
                s = self._make_system(u)
                total_before = sum(t.utilization for t in s.tasks)
                unbalance_contended(s)
                total_after = sum(t.utilization for t in s.tasks)
                self.assertAlmostEqual(total_before, total_after,
                                       msg=f"Total utilisation changed at u={u}")

    # -- contention --

    def test_creates_contention(self):
        """At least one processor should be significantly above average."""
        s = self._make_system(0.7)
        unbalance_contended(s)
        loads, _ = self._processor_loads(s)
        avg = sum(loads) / len(loads)
        max_load = max(loads)
        self.assertGreater(max_load, avg * 1.1,
                           f"max load {max_load:.3f} not > 1.1 * avg {avg:.3f}")

    def test_creates_cold_processor(self):
        """At high enough utilisation, at least one processor is below average."""
        s = self._make_system(0.7)
        unbalance_contended(s)
        loads, _ = self._processor_loads(s)
        avg = sum(loads) / len(loads)
        min_load = min(loads)
        self.assertLess(min_load, avg * 0.9,
                        f"min load {min_load:.3f} not < 0.9 * avg {avg:.3f}")

    # -- edge cases --

    def test_raises_when_avg_ge_max(self):
        s = self._make_system(0.96)
        with self.assertRaises(ValueError):
            unbalance_contended(s, max_utilization=0.95)

    def test_low_utilization_still_has_valid_loads(self):
        s = self._make_system(0.3)
        unbalance_contended(s, max_utilization=0.95)
        loads, _ = self._processor_loads(s)
        for load in loads:
            self.assertLess(load, 0.95)


if __name__ == '__main__':
    unittest.main()
