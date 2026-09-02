from gradient_descent.interfaces import StopFunction
from model.system_model import SystemModel


import time


class ThresholdStopFunction(StopFunction):
    def __init__(self, limit=100, threshold=0, patience=None, max_time=None):
        self.limit = limit
        self.threshold = threshold
        self.patience = patience
        self.max_time = max_time
        self.best = float("inf")  # best cost value
        self.xb = None            # best solution
        self.stalled = 0          # iterations without improvement
        self._start = time.perf_counter()

    def reset(self):
        self.best = float("inf")
        self.xb = None
        self.stalled = 0
        self._start = time.perf_counter()

    def should_stop(self, S: SystemModel, x: [float], cost: float, t: int) -> bool:
        if cost < self.best:
            self.best = cost
            self.xb = x
            self.stalled = 0
        else:
            self.stalled += 1
        if self.max_time is not None and time.perf_counter() - self._start > self.max_time:
            return True
        return cost < self.threshold or t > self.limit or (
            self.patience is not None and self.stalled > self.patience)

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