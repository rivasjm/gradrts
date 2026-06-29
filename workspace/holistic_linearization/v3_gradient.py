"""GradientFunction backed by differentiable V3 soft-priority surrogate.

Provides the gradient of avg-WCRT w.r.t. per-task priority scores
computed via PyTorch autograd, **without** finite differences.
"""

import numpy as np
import torch

from gradient_descent.interfaces import GradientFunction
from model.linear_system import LinearSystem
from workspace.holistic_linearization.differentiable_v3 import v3_soft_priority


class V3SoftPriorityGradient(GradientFunction):
    """Gradient of avg-WCRT w.r.t. task priorities via V3 autograd.

    Parameters
    ----------
    tau : float
        Temperature for the sigmoid that softens priority ordering.
        Lower = sharper (closer to hard priority).  Default 0.5.
    """

    def __init__(self, tau=0.5):
        self.tau = tau
        self._n_calls = 0

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        """Return gradient of average flow WCRT w.r.t. *x*.

        *x* must be the flat priority vector (one float per task).
        The values are inserted as ``task.priority = x[i]`` before the
        V3 model is built.
        """
        self._n_calls += 1

        tasks = system.tasks
        n = len(tasks)
        t2i = {t: i for i, t in enumerate(tasks)}

        # --- insert x into system so task.priority reflects current x ---
        for t, xi in zip(tasks, x):
            t.priority = float(xi)

        # --- build predicate arrays ---
        pred_idx = [-1] * n
        for i, t in enumerate(tasks):
            p = t.predecessors
            pred_idx[i] = t2i[p[0]] if p else -1

        same_proc = torch.zeros(n, n, dtype=torch.bool)
        for i, ti in enumerate(tasks):
            for j, tj in enumerate(tasks):
                same_proc[i, j] = (ti.processor == tj.processor)

        C = torch.tensor([t.wcet for t in tasks], dtype=torch.float64)
        T = torch.tensor([t.period for t in tasks], dtype=torch.float64)

        # --- V3 soft-priority forward (with scores from current x) ---
        s_init = torch.tensor(x, dtype=torch.float64)
        r, s_leaf = v3_soft_priority(C, T, pred_idx, same_proc, tau=self.tau, s_init=s_init)

        # --- cost = avg WCRT of flow-final tasks ---
        last_idx = [i for i, t in enumerate(tasks) if t.is_last]
        avg_wcrt = r[last_idx].mean()
        avg_wcrt.backward()

        grad_s = s_leaf.grad
        if grad_s is None:
            return [0.0] * n

        g = grad_s.detach().numpy()
        if np.any(~np.isfinite(g)):
            return [0.0] * n

        return g.tolist()
