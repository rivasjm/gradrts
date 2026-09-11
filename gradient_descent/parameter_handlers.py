from gradient_descent.interfaces import ParameterHandler
from model.linear_system import LinearSystem
import math


class DeadlineHandler(ParameterHandler):
    def extract(self, system: LinearSystem) -> [float]:
        max_d = max([flow.deadline for flow in system.flows])
        x = [sigmoid(t.deadline / max_d) for t in system.tasks]
        return x

    def insert(self, system: LinearSystem, x: [float]):
        max_d = max([flow.deadline for flow in system.flows])
        tasks = system.tasks
        assert len(tasks) == len(x)
        for v, t in zip(x, system.tasks):
            t.deadline = v * max_d                   


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


class FPMappingHandler(ParameterHandler):
    def __init__(self):
        self.fp_handler = FPHandler()

    def reset(self):
        self.fp_handler.reset()

    def extract(self, S: LinearSystem) -> [float]:
        m_vector = [0.55 if task.processor == proc else 0.45 for task in S.tasks for proc in S.processors]
        p_vector = self.fp_handler.extract(S)
        return m_vector + p_vector

    def insert(self, S: LinearSystem, x: [float]) -> None:
        tasks = S.tasks
        procs = S.processors
        p = len(procs)
        t = len(tasks)
        assert len(x) == p*t + t

        # parse mapping values (fist p*t values)
        for i in range(t):
            sub = x[i*p: (i+1)*p]
            proc_index = sub.index(max(sub))
            tasks[i].processor = procs[proc_index]

        # parse priority values (last t values)
        self.fp_handler.insert(S, x[-t:])

    def mapping_mask(self, S: LinearSystem) -> [bool]:
        """True for the mapping block (first p*t coordinates), False for priorities."""
        p = len(S.processors)
        t = len(S.tasks)
        return [True] * (p * t) + [False] * t


class DeadlineMappingHandler(ParameterHandler):
    def __init__(self):
        self.deadline_handler = DeadlineHandler()

    def reset(self):
        self.deadline_handler.reset()

    def extract(self, S: LinearSystem) -> [float]:
        m_vector = [0.55 if task.processor == proc else 0.45 for task in S.tasks for proc in S.processors]
        t_vector = self.deadline_handler.extract(S)
        return m_vector + t_vector

    def insert(self, S: LinearSystem, x: [float]) -> None:
        tasks = S.tasks
        procs = S.processors
        p = len(procs)
        t = len(tasks)
        assert len(x) == p*t + t

        # parse mapping values (fist p*t values)
        for i in range(t):
            sub = x[i*p: (i+1)*p]
            proc_index = sub.index(max(sub))
            tasks[i].processor = procs[proc_index]

        # parse priority values (last t values)
        self.deadline_handler.insert(S, x[-t:])

    def mapping_mask(self, S: LinearSystem) -> [bool]:
        """True for the mapping block (first p*t coordinates), False for deadlines."""
        p = len(S.processors)
        t = len(S.tasks)
        return [True] * (p * t) + [False] * t


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

        from model.analysis_function import calculate_priorities, normalize_priorities
        calculate_priorities(S)
        normalize_priorities(S)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))