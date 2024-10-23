from gradient_descent.interfaces import StopFunction
from model.system_model import SystemModel


class ThresholdStopFunction(StopFunction):
    def __init__(self, limit=100, threshold=0):
        self.limit = limit
        self.threshold = threshold
        self.best = float("inf")  # best cost value
        self.xb = None            # best solution

    def reset(self):
        self.best = float("inf")
        self.xb = None

    def should_stop(self, S: SystemModel, x: [float], cost: float, t: int) -> bool:
        if cost < self.best:
            self.best = cost
            self.xb = x
        return cost < self.threshold or t > self.limit

    def solution(self, S: SystemModel):
        return self.xb

    def solution_cost(self):
        return self.best


class FixedIterationsStop(StopFunction):
    def __init__(self, iterations=100):
        self.iterations = iterations
        self.best = float("inf")  # best cost value
        self.xb = None            # best solution

    def reset(self):
        self.best = float("inf")
        self.xb = None

    def should_stop(self, S: SystemModel, x: [float], cost: float, t: int) -> bool:
        if cost < self.best:
            self.best = cost
            self.xb = x
        return t > self.iterations

    def solution(self, S: SystemModel):
        return self.xb

    def solution_cost(self):
        return self.best