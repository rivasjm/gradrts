from gradient_descent.interfaces import CostFunction, ParameterHandler
from model.linear_system import LinearSystem
from model.linear_system_utils import backup_assignment, restore_assignment
from model.analysis_function import AnalysisFunction


class InvslackCost(CostFunction):
    def __init__(self, param_handler: ParameterHandler, analysis: AnalysisFunction):
        self.param_handler = param_handler
        self.analysis = analysis

    def reset(self):
        self.param_handler.reset()

    def compute(self, S: LinearSystem, x: [float]) -> float:
        a = backup_assignment(S)
        self.param_handler.insert(S, x)
        self.analysis.apply(S)
        cost = max([(flow.wcrt - flow.deadline) / flow.deadline for flow in S.flows])
        restore_assignment(S, a)
        return cost