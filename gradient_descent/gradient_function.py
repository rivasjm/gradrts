from gradient_descent.interfaces import GradientFunction, CostFunction
from model.analysis_function import Function
from model.system_model import SystemModel


class SequentialGradientFunction(GradientFunction):
    def __init__(self, cost_function: CostFunction, sigma=1.5):
        self.delta_function = AvgSeparationDelta(sigma=sigma)
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
    def __init__(self, sigma=1.5):
        self.sigma = sigma

    def apply(self, S: SystemModel, x: [float]) -> [float]:
        seps = [abs(x[i + 1] - x[i]) for i in range(len(x) - 1)]
        return [self.sigma * sum(seps) / len(seps)]*len(x)


class BlockConstantDelta(AvgSeparationDelta):
    """AvgSeparationDelta with a per-block override for the finite-difference step.

    ``blocks`` is a sequence of ``(count, delta)`` pairs describing consecutive
    parameter blocks (in the same order as the parameter vector); a ``delta``
    of ``None`` keeps the shared delta for that block. Coordinates after the
    last block also keep the shared delta, so a partial description is allowed.
    This lets the blocks be steered independently (e.g. a larger step for a
    mapping block without disturbing a priority block).
    """

    def __init__(self, sigma=1.5, blocks=()):
        super().__init__(sigma=sigma)
        self.blocks = list(blocks)

    def apply(self, S: SystemModel, x: [float]) -> [float]:
        base = super().apply(S, x)
        out = []
        i = 0
        for count, delta in self.blocks:
            for _ in range(count):
                out.append(base[i] if delta is None else delta)
                i += 1
        out.extend(base[i:])
        return out


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