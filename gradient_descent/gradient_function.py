from gradient_descent.interfaces import GradientFunction, CostFunction
from model.analysis_function import Function
from model.system_model import SystemModel


class SequentialGradientFunction(GradientFunction):
    def __init__(self, cost_function: CostFunction, lambda_factor=1.5):
        self.delta_function = AvgSeparationDelta(factor=lambda_factor)
        self.cost_function = cost_function

    def reset(self):
        self.delta_function.reset()
        self.cost_function.reset()

    def compute(self, S: SystemModel, x: [float]) -> [float]:
        deltas = self.delta_function.apply(S, x)
        inputs = gradient_inputs_from_deltas(x, deltas)
        costs = [self.cost_function.compute(S, x) for x in inputs]
        gradient = gradient_from_costs(costs, deltas)
        return gradient


class AvgSeparationDelta(Function):
    def __init__(self, factor=1.5):
        self.factor = factor

    def apply(self, S: SystemModel, x: [float]) -> [float]:
        seps = [abs(x[i + 1] - x[i]) for i in range(len(x) - 1)]
        return [self.factor * sum(seps) / len(seps)]*len(x)


def gradient_inputs_from_deltas(x, deltas) -> [[float]]:
    ret = []
    for i in range(len(x)):
        vector = x[:]
        vector[i] += deltas[i]
        ret.append(vector)
        vector = x[:]
        vector[i] -= deltas[i]
        ret.append(vector)
    return ret

def gradient_from_costs(costs, deltas) -> [float]:
    gradient = [0] * int(len(costs) / 2)
    for i in range(len(gradient)):
        gradient[i] = (costs[2*i] - costs[2*i + 1]) / \
                      (2 * deltas[i % len(deltas)])
    return gradient