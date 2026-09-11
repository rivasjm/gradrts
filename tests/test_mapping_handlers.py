import unittest

from examples.examples_special import get_validation_example
from gradient_descent.parameter_handlers import DeadlineMappingHandler, FPMappingHandler


class MappingMaskTest(unittest.TestCase):
    def setUp(self):
        self.system = get_validation_example()
        self.n_procs = len(self.system.processors)
        self.n_tasks = len(self.system.tasks)
        self.mapping_block = self.n_procs * self.n_tasks

    def _assert_mask(self, handler):
        mask = handler.mapping_mask(self.system)
        self.assertEqual(len(mask), self.mapping_block + self.n_tasks)
        self.assertEqual(len(mask), len(handler.extract(self.system)))
        self.assertTrue(all(mask[:self.mapping_block]))
        self.assertFalse(any(mask[self.mapping_block:]))

    def test_fp_mapping_mask(self):
        self._assert_mask(FPMappingHandler())

    def test_deadline_mapping_mask(self):
        self._assert_mask(DeadlineMappingHandler())


if __name__ == '__main__':
    unittest.main()
