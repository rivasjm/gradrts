from gradient_descent.interfaces import ParameterHandler
from model.linear_system import LinearSystem
import math


class DeadlineHandler(ParameterHandler):
    """Normalized local-deadline parameters in [0, 1].

    ``extract`` and ``insert`` are exact inverses relative to the largest flow
    deadline, so the optimizer starts from the assigned (e.g. PD) deadlines
    instead of a distorted version of them. Values outside [0, 1] produced by
    the updates are clamped, which bounds the search to valid deadlines.
    """

    def extract(self, system: LinearSystem) -> [float]:
        max_d = max([flow.deadline for flow in system.flows])
        return [t.deadline / max_d for t in system.tasks]

    def insert(self, system: LinearSystem, x: [float]):
        max_d = max([flow.deadline for flow in system.flows])
        tasks = system.tasks
        assert len(tasks) == len(x)
        for v, t in zip(x, tasks):
            t.deadline = min(max(v, 0.0), 1.0) * max_d


class FPHandler(ParameterHandler):
    def extract(self, system: LinearSystem) -> [float]:
        # max_priority = max(map(lambda t: t.priority, system.tasks))
        r = [sigmoid(t.priority) for t in system.tasks]
        return r

    def insert(self, system: LinearSystem, x: [float]):
        tasks = system.tasks
        assert len(tasks) == len(x)
        for v, t in zip(x, tasks):
            t.priority = v


class CompoundHandler(ParameterHandler):
    """A handler whose parameter vector is the concatenation of its components'.

    Each component handler owns a contiguous slice of the vector; the compound
    knows nothing about what the components mean, it just concatenates their
    ``extract`` results and splits the vector back on ``insert``. The block
    layout is therefore exactly the components' sizes, in the given order.
    """

    def __init__(self, handlers):
        self.handlers = list(handlers)

    def reset(self):
        for handler in self.handlers:
            handler.reset()

    def block_sizes(self, S: LinearSystem) -> [int]:
        return [len(handler.extract(S)) for handler in self.handlers]

    def extract(self, S: LinearSystem) -> [float]:
        return [v for handler in self.handlers for v in handler.extract(S)]

    def insert(self, S: LinearSystem, x: [float]) -> None:
        sizes = self.block_sizes(S)
        if len(x) != sum(sizes):
            raise ValueError(f"expected {sum(sizes)} parameters, got {len(x)}")
        i = 0
        for handler, size in zip(self.handlers, sizes):
            handler.insert(S, x[i:i + size])
            i += size

    def block_mask(self, S: LinearSystem, handler: ParameterHandler) -> [bool]:
        """Boolean mask selecting the coordinates owned by ``handler``.

        Raises if ``handler`` is not one of this compound's components.
        """
        if not any(h is handler for h in self.handlers):
            raise ValueError("handler is not part of this compound")
        mask = []
        for h in self.handlers:
            mask.extend([h is handler] * len(h.extract(S)))
        return mask


class MappingHandler(ParameterHandler):
    def extract(self, S: LinearSystem) -> [float]:
        return [0.55 if task.processor == proc else 0.45
                for task in S.tasks for proc in S.processors]

    def insert(self, S: LinearSystem, x: [float]) -> None:
        tasks = S.tasks
        procs = S.processors
        p = len(procs)
        t = len(tasks)
        assert len(x) == p * t
        for i in range(t):
            sub = x[i * p: (i + 1) * p]
            proc_index = sub.index(max(sub))
            tasks[i].processor = procs[proc_index]


class FPMappingHandler(CompoundHandler):
    """Mapping block followed by a fixed-priority block (see CompoundHandler)."""

    def __init__(self):
        self.fp_handler = FPHandler()
        super().__init__([MappingHandler(), self.fp_handler])

    def mapping_mask(self, S: LinearSystem) -> [bool]:
        """True for the mapping block (first p*t coordinates), False for priorities."""
        return self.block_mask(S, self.handlers[0])


class DeadlineMappingHandler(CompoundHandler):
    """Mapping block followed by a deadline block (see CompoundHandler)."""

    def __init__(self):
        self.deadline_handler = DeadlineHandler()
        super().__init__([MappingHandler(), self.deadline_handler])

    def mapping_mask(self, S: LinearSystem) -> [bool]:
        """True for the mapping block (first p*t coordinates), False for deadlines."""
        return self.block_mask(S, self.handlers[0])


def sigmoid(x):
    return 1 / (1 + math.exp(-x))