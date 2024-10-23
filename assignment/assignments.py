from random import Random

from model.analysis_function import AnalysisFunction, globalize_deadlines, calculate_priorities, normalize_priorities
from model.linear_system import LinearSystem
from utils.exec_time import ExecTime


class PDAssignment(AnalysisFunction):
    def __init__(self, normalize=False, globalize=False):
        self.normalize = normalize
        self.globalize = globalize
        self.exec_time = ExecTime()

    def apply(self, system: LinearSystem):
        self.exec_time.init()
        self.calculate_local_deadlines(system)
        if self.globalize:
            globalize_deadlines(system)
        calculate_priorities(system)
        if self.normalize:
            normalize_priorities(system)
        self.exec_time.stop()
        return system

    @staticmethod
    def calculate_local_deadlines(system):
        for flow in system:
            sum_wcet = sum(map(lambda t: t.wcet, flow.tasks))
            for task in flow:
                d = task.wcet * flow.deadline / sum_wcet
                task.deadline = d


class PassthroughAssignment:
    def __init__(self, normalize=False):
        self.normalize = normalize

    def apply(self, system: LinearSystem):
        if self.normalize:
            normalize_priorities(system)


class RandomAssignment:
    def __init__(self, random=Random(42), normalize=False):
        self.random = random
        self.normalize = normalize

    def apply(self, system: LinearSystem):
        tasks = system.tasks
        self.random.shuffle(tasks)
        for task, priority in zip(tasks, range(1, len(tasks)+1)):
            task.priority = priority
        if self.normalize:
            normalize_priorities(system)


class EQSAssignment:
    def apply(self, system: LinearSystem):
        self.compute_deadlines(system)
        calculate_priorities(system)

    @staticmethod
    def compute_deadlines(system: LinearSystem):
        for flow in system:
            s = 0
            n = len(flow.tasks)
            for j in reversed(range(len(flow.tasks))):
                task = flow[j]
                s += task.wcet
                task.deadline = task.wcet + (flow.deadline - s)/(n - (j+1) + 1)


class EQFAssignment:
    def apply(self, system: LinearSystem):
        self.compute_deadlines(system)
        calculate_priorities(system)

    @staticmethod
    def compute_deadlines(system: LinearSystem):
        for flow in system:
            s = 0
            n = len(flow.tasks)
            for j in reversed(range(len(flow.tasks))):
                task = flow[j]
                s += task.wcet
                task.deadline = task.wcet + (flow.deadline-s)*(task.wcet/s)