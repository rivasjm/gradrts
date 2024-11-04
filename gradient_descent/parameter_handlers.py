from gradient_descent.interfaces import ParameterHandler
from model.linear_system import LinearSystem
import math


class DeadlineExtractor(ParameterHandler):
    def extract(self, system: LinearSystem) -> [float]:
        max_d = max([task.deadline for task in system.tasks])
        x = [sigmoid(t.deadline/max_d) for t in system.tasks]
        return x

    def insert(self, system: LinearSystem, x: [float]):
        max_d = max([task.deadline for task in system.tasks])
        tasks = system.tasks
        assert len(tasks) == len(x)
        for v, t in zip(x, tasks):
            t.deadline = v*max_d


class PriorityExtractor(ParameterHandler):
    def extract(self, system: LinearSystem) -> [float]:
        # max_priority = max(map(lambda t: t.priority, system.tasks))
        r = [sigmoid(t.priority) for t in system.tasks]
        return r

    def insert(self, system: LinearSystem, x: [float]):
        tasks = system.tasks
        assert len(tasks) == len(x)
        for v, t in zip(x, tasks):
            t.priority = v


class MappingPriorityExtractor(ParameterHandler):
    def __init__(self):
        self.prio_extractor = PriorityExtractor()

    def reset(self):
        self.prio_extractor.reset()

    def extract(self, S: LinearSystem) -> [float]:
        m_vector = [0.55 if task.processor == proc else 0.45 for task in S.tasks for proc in S.processors]
        p_vector = self.prio_extractor.extract(S)
        return m_vector + p_vector

    def insert(self, S: LinearSystem, x: [float]) -> None:
        tasks = S.tasks
        procs = S.processors
        p = len(procs)
        t = len(tasks)
        assert len(x) == p*t + t

        # parse mapping values (fist p*t values)
        for i in range(t):
            sub = x[i*p: i*p+3]
            proc_index = sub.index(max(sub))
            tasks[i].processor = procs[proc_index]

        # parse priority values (last t values)
        self.prio_extractor.insert(S, x[-t:])


class MappingDeadlineExtractor(ParameterHandler):
    def __init__(self):
        self.deadline_extractor = DeadlineExtractor()

    def reset(self):
        self.deadline_extractor.reset()

    def extract(self, S: LinearSystem) -> [float]:
        m_vector = [0.55 if task.processor == proc else 0.45 for task in S.tasks for proc in S.processors]
        t_vector = self.deadline_extractor.extract(S)
        return m_vector + t_vector

    def insert(self, S: LinearSystem, x: [float]) -> None:
        tasks = S.tasks
        procs = S.processors
        p = len(procs)
        t = len(tasks)
        assert len(x) == p*t + t

        # parse mapping values (fist p*t values)
        for i in range(t):
            sub = x[i*p: i*p+3]
            proc_index = sub.index(max(sub))
            tasks[i].processor = procs[proc_index]

        # parse priority values (last t values)
        self.deadline_extractor.insert(S, x[-t:])


def sigmoid(x):
    return 1 / (1 + math.exp(-x))