import unittest

from examples.examples_special import get_validation_example
from gradient_descent.parameter_handlers import (CompoundHandler, DeadlineHandler,
                                                 FPHandler, MappingHandler)


class CompoundHandlerTest(unittest.TestCase):
    def setUp(self):
        self.system = get_validation_example()
        self.n_procs = len(self.system.processors)
        self.n_tasks = len(self.system.tasks)
        self.mapping_block = self.n_procs * self.n_tasks

    def _assert_mapping_mask(self, other):
        mapping = MappingHandler()
        handler = CompoundHandler([mapping, other])
        mask = handler.block_mask(self.system, mapping)
        self.assertEqual(len(mask), self.mapping_block + self.n_tasks)
        self.assertEqual(len(mask), len(handler.extract(self.system)))
        self.assertTrue(all(mask[:self.mapping_block]))
        self.assertFalse(any(mask[self.mapping_block:]))

    def test_fp_mapping_mask(self):
        self._assert_mapping_mask(FPHandler())

    def test_deadline_mapping_mask(self):
        self._assert_mapping_mask(DeadlineHandler())

    def test_block_sizes_and_size(self):
        mapping = MappingHandler()
        fp = FPHandler()
        handler = CompoundHandler([mapping, fp])
        self.assertEqual(handler.block_sizes(self.system),
                         [self.mapping_block, self.n_tasks])
        self.assertEqual(handler.size(self.system),
                         self.mapping_block + self.n_tasks)
        self.assertEqual(handler.size(self.system),
                         len(handler.extract(self.system)))

    def test_block_mask_rejects_foreign_handler(self):
        handler = CompoundHandler([MappingHandler(), FPHandler()])
        with self.assertRaises(ValueError):
            handler.block_mask(self.system, MappingHandler())


if __name__ == '__main__':
    unittest.main()
