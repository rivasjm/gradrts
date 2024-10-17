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