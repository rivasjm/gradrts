"""V1-FD: Finite-difference gradient using V1 iterative analysis.

Replaces the Holistic FP in the gradient computation with V1
(linearised iterative, Spearman 0.94), making each perturbation
evaluation ~20x faster while keeping the gradient direction
well-aligned.
"""

from gradient_descent.interfaces import GradientFunction
from model.linear_system import LinearSystem, Task
from model.analysis_function import init_wcrt, reset_wcrt

from workspace.holistic_linearization.linearized_fp import _higher_priority


def v1_analyse(system: LinearSystem, limit_factor=10, max_p=100) -> float:
    """Run hard-priority V1 iterative analysis and return avg flow WCRT.

    Mutates ``task.wcrt`` in-place.  Returns ``float('inf')`` on failure.
    """
    reset_wcrt(system)
    init_wcrt(system)

    tasks = system.tasks
    wcrts = [t.wcrt for t in tasks]
    wcrts_prev = [0.0] * len(tasks)

    while wcrts != wcrts_prev:
        wcrts_prev = wcrts[:]

        for task in tasks:
            hp = _higher_priority(task)
            limit = task.flow.deadline * limit_factor
            T_i = task.period
            C_i = task.wcet
            J_i = task.jitter

            U_hp = sum(t.wcet / t.period for t in hp)
            denominator = 1.0 - U_hp
            if denominator <= 0:
                return float('inf')

            JU_hp = sum(t.jitter * t.wcet / t.period for t in hp)

            p = 1
            while p <= max_p:
                w = (p * C_i + JU_hp) / denominator
                r = w - (p - 1) * T_i + J_i

                if r > task.wcrt:
                    task.wcrt = r

                if r > limit:
                    return float('inf')

                if w <= p * T_i:
                    break

                p += 1

        wcrts = [t.wcrt for t in tasks]

    wcrts = [f.wcrt for f in system.flows if f.wcrt is not None]
    return sum(wcrts) / len(wcrts) if wcrts else float('inf')


class V1FiniteDifferenceGradient(GradientFunction):
    """Central finite-difference gradient using V1 surrogate.

    Perturbs the *x* vector (sigmoid-squashed priorities, as produced by
    ``PriorityExtractor``) by ``±eps``, runs V1 analysis on each perturbed
    configuration, and returns the gradient in *x*-space.

    Parameters
    ----------
    eps : float
        Perturbation in x-space (default 0.05).
    limit_factor : float
        Limit factor for V1 analysis.
    """

    def __init__(self, eps=0.05, limit_factor=10):
        self.eps = eps
        self.limit_factor = limit_factor

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        from gradient_descent.parameter_handlers import PriorityExtractor

        tasks = system.tasks
        n = len(tasks)
        grad = [0.0] * n

        extractor = PriorityExtractor()

        # Insert base x
        extractor.insert(system, x)

        f0 = v1_analyse(system, limit_factor=self.limit_factor)
        if not (f0 < float('inf')):
            return grad

        for i in range(n):
            x_plus = list(x)
            x_minus = list(x)
            x_plus[i] = min(x[i] + self.eps, 1.0 - 1e-6)
            x_minus[i] = max(x[i] - self.eps, 1e-6)

            extractor.insert(system, x_plus)
            fp = v1_analyse(system, limit_factor=self.limit_factor)

            extractor.insert(system, x_minus)
            fm = v1_analyse(system, limit_factor=self.limit_factor)

            if fp < float('inf') and fm < float('inf'):
                grad[i] = (fp - fm) / (2.0 * self.eps)

        # Restore
        extractor.insert(system, x)
        return grad
