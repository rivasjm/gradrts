from gradient_descent.interfaces import CostFunction, ParameterHandler
from model.linear_system import LinearSystem
from model.linear_system_utils import backup_assignment, restore_assignment
from model.analysis_function import AnalysisFunction


class InvslackCost(CostFunction):
    def __init__(self, parameter_handler: ParameterHandler, analysis: AnalysisFunction):
        self.parameter_handler = parameter_handler
        self.analysis = analysis

    def reset(self):
        self.parameter_handler.reset()

    def compute(self, S: LinearSystem, x: [float]) -> float:
        a = backup_assignment(S)
        self.parameter_handler.insert(S, x)
        self.analysis.apply(S)
        cost = max([(flow.wcrt - flow.deadline) / flow.deadline for flow in S.flows])
        restore_assignment(S, a)
        return cost