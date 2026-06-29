"""GradientFunction implementations using the differentiable V1 surrogate.

- ``V1UnrolledGradient``  — Phase 1: N unrolled iterations
- ``V1UnrolledAnnealedGradient`` — Phase 2: same as above + τ schedule
- ``V1ImplicitGradient``  — Phase 3: implicit diff at fixed point
"""

import numpy as np
import torch

from gradient_descent.interfaces import GradientFunction
from model.linear_system import LinearSystem
from workspace.holistic_linearization.differentiable_v1 import (
    v1_unrolled_forward,
    v1_implicit_forward,
    exponential_decay,
)


def _compute_gradient(system, x, forward_fn, **kwargs):
    """Shared logic: forward fn → avg_wcrt → backward → grad."""
    tasks = system.tasks
    n = len(tasks)
    dtype = torch.float64

    s_init = torch.tensor(x, dtype=dtype)

    try:
        r, s_leaf = forward_fn(system, s_init=s_init, **kwargs)
    except Exception:
        return [0.0] * n

    last_idx = [i for i, t in enumerate(tasks) if t.is_last]
    avg = r[last_idx].mean()
    avg.backward()

    g = s_leaf.grad
    if g is None:
        return [0.0] * n

    grad = g.detach().numpy()
    if np.any(~np.isfinite(grad)):
        return [0.0] * n

    gn = np.linalg.norm(grad)
    if gn > 1.0:
        grad = grad / gn

    return grad.tolist()


class V1UnrolledGradient(GradientFunction):
    """Phase 1: Unrolled V1 gradient (N=10, tau=0.5)."""

    def __init__(self, tau=0.5, N=10):
        self.tau = tau
        self.N = N

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        return _compute_gradient(system, x, v1_unrolled_forward,
                                 tau=self.tau, N=self.N)


class V1UnrolledAnnealedGradient(GradientFunction):
    """Phase 2: Unrolled V1 with annealing τ(t).

    τ starts at tau_0 and decays each call to the gradient function.
    """

    def __init__(self, tau_0=2.0, tau_min=0.05, decay=0.95, N=10):
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.decay = decay
        self.N = N
        self._t = 0
        self._schedule = exponential_decay(tau_0, tau_min, decay)

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        tau = self._schedule(self._t)
        self._t += 1
        return _compute_gradient(system, x, v1_unrolled_forward,
                                 tau=tau, N=self.N)


class V1ImplicitGradient(GradientFunction):
    """Phase 3: Implicit differentiation at V1 fixed point (tau=0.5)."""

    def __init__(self, tau=0.5, max_iters=200, tol=1e-8):
        self.tau = tau
        self.max_iters = max_iters
        self.tol = tol

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        return _compute_gradient(system, x, v1_implicit_forward,
                                 tau=self.tau, max_iters=self.max_iters,
                                 tol=self.tol)
