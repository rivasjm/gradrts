from gradient_descent.interfaces import ParameterHandler, CostFunction, StopFunction, GradientFunction, UpdateFunction
from model.analysis_function import Function
from model.system_model import SystemModel


class GradientDescentOptimizer(Function):
    def __init__(self,
                 parameter_handler: ParameterHandler,
                 cost_function: CostFunction,               # function to evaluate the cost at each iteration
                 stop_function: StopFunction,               # function to determine when to stop the optimization
                 gradient_function: GradientFunction,       # function to compute the gradient
                 update_function: UpdateFunction,           # function to compute update vector from a gradient
                 ref_cost_function: CostFunction = None,    # (optional) secondary cost function for logging
                 callback = None,                           # (optional) callback that is called each iteration
                 verbose = False,                           # verbose flag
                 chunk: int = 0,                            # iterations per restart (0 disables restarts)
                 restart_x: [float] = None                  # restart point (None = the initial input)
                 ):
        self.parameter_handler = parameter_handler
        self.cost_function = cost_function
        self.stop_function = stop_function
        self.gradient_function = gradient_function
        self.update_function = update_function
        self.ref_cost_function = ref_cost_function
        self.callback = callback
        self.verbose = verbose
        self.chunk = chunk
        self.restart_x = restart_x

    def reset(self):
        self.parameter_handler.reset()
        self.cost_function.reset()
        self.stop_function.reset()
        self.gradient_function.reset()
        self.update_function.reset()
        self.ref_cost_function.reset()

    def apply(self, S: SystemModel) -> [float]:
        t = 1                   # global iteration count (for the stop function)
        local_t = 1             # iteration count within the current chunk
        x = self.parameter_handler.extract(S)  # initial input
        restart_x = list(x) if self.restart_x is None else list(self.restart_x)
        best = float('inf')     # best cost value, for logging purposes, not necessarily the cost of the solution
        ref_cost = None         # optional alternative cost value, just for logging purposes
        xb = x                  # best input, for logging purposes, not necessarily returned as solution

        while True:
            cost = self.cost_function.compute(S, x)
            if cost < best:
                best = cost
                xb = x

            if self.ref_cost_function:
                ref_cost = self.ref_cost_function.compute(S, x)

            if self.verbose:
                msg = f"iteration={t}: cost={cost:.3f} best={best:.3f}"
                if self.ref_cost_function:
                    msg += f" ref={ref_cost:.3f}"
                print(msg)

            if self.callback:
                self.callback(t, S, x, xb, cost, best, ref_cost)

            stop = self.stop_function.should_stop(S, x, cost, t)
            if stop:
                break

            nabla = self.gradient_function.compute(S, x)
            # the update function sees the local iteration count, so its noise
            # schedule and warmup start over with every chunk
            update = self.update_function.update(S, x, nabla, local_t)
            x = [a + b for a, b in zip(x, update)]
            t = t + 1

            # insert into system, extract again to get x properly normalized
            self.parameter_handler.insert(S, x)
            x = self.parameter_handler.extract(S)

            # after ``chunk`` local iterations, restart from the same point with
            # a fresh update-function seed derived from the chunk index
            if self.chunk and local_t == self.chunk:
                x = list(restart_x)
                self.parameter_handler.insert(S, x)
                self.update_function.reset(seed=t // self.chunk)
                local_t = 1
            else:
                local_t += 1

        solution = self.stop_function.solution(S)
        self.parameter_handler.insert(S, solution)
        if self.verbose:
            print(f"Returning solution with cost={self.stop_function.solution_cost():.3f}")

        return solution