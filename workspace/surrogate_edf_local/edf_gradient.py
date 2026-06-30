"""GradientFunction implementations using the differentiable EDF surrogate."""

import numpy as np
import torch

from gradient_descent.interfaces import GradientFunction
from model.linear_system import LinearSystem
from workspace.surrogate_edf_local.differentiable_edf import (
    edf_surrogate_forward,
    _build_system_tensors,
)


class SurrogateEDFGradient(GradientFunction):
    """Gradient of the EDF schedulability cost via the differentiable surrogate.

    Uses unrolled fixed-point iterations (the surrogate forward pass)
    followed by automatic differentiation to obtain the gradient of the
    cost w.r.t. sigmoid-encoded deadlines.

    Parameters
    ----------
    tau : float
        Temperature for soft ceil / floor / comparisons.
    N_w : int
        Unrolled iterations for the *w_ab* fixed point.
    N_jitter : int
        Outer jitter-update passes.
    M_psi : int
        Number of psi grid points per job p.
    temperature_max : float
        Temperature for the soft-max over (p, psi).
    grad_clip : float or None
        If set, clip gradient norm to this value.
    """

    def __init__(
        self,
        tau: float = 0.5,
        N_w: int = 10,
        N_jitter: int = 2,
        M_psi: int = 50,
        temperature_max: float = 0.1,
        grad_clip: float = 1.0,
    ):
        self.tau = tau
        self.N_w = N_w
        self.N_jitter = N_jitter
        self.M_psi = M_psi
        self.temperature_max = temperature_max
        self.grad_clip = grad_clip

    # ------------------------------------------------------------------
    # GradientFunction interface
    # ------------------------------------------------------------------

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        tensors = _build_system_tensors(system)
        last_idx = tensors[5]  # 6th element: list of last-task indices
        n = len(system.tasks)
        dtype = torch.float64

        s = torch.tensor(x, dtype=dtype, requires_grad=True)

        try:
            r, s_leaf = edf_surrogate_forward(
                system,
                s,
                tau=self.tau,
                N_w=self.N_w,
                N_jitter=self.N_jitter,
                M_psi=self.M_psi,
                temperature_max=self.temperature_max,
            )
        except Exception:
            return [0.0] * n

        # Cost: smooth approximation of max relative slack of last tasks
        # (same idea as InvslackCost but differentiable everywhere)
        if not last_idx:
            return [0.0] * n

        r_last = r[last_idx]
        # deadlines for last tasks (decode sigmoid)
        max_d = max(t.deadline for t in system.tasks) if system.tasks else 1.0
        D_last = s_leaf[last_idx] * max_d

        # slack = (r - D) / D  for each flow's last task
        slack = (r_last - D_last) / D_last.clamp(min=1e-9)

        # smooth max via logsumexp (differentiable, works for negative slacks too)
        cost = torch.logsumexp(slack / self.temperature_max, dim=0) * self.temperature_max

        cost.backward()

        g = s_leaf.grad
        if g is None:
            return [0.0] * n

        grad = g.detach().numpy()

        # Handle non-finite
        if np.any(~np.isfinite(grad)):
            return [0.0] * n

        # Clip gradient norm
        if self.grad_clip is not None:
            gn = np.linalg.norm(grad)
            if gn > self.grad_clip:
                grad = grad * (self.grad_clip / gn)

        return grad.tolist()


class SurrogateEDFAnnealedGradient(SurrogateEDFGradient):
    """Same surrogate but with an annealing schedule for tau.

    tau starts at ``tau_0`` and decays by factor ``decay`` each
    gradient call, down to ``tau_min``.
    """

    def __init__(
        self,
        tau_0: float = 2.0,
        tau_min: float = 0.05,
        decay: float = 0.95,
        N_w: int = 10,
        N_jitter: int = 2,
        M_psi: int = 50,
        temperature_max: float = 0.1,
        grad_clip: float = 1.0,
    ):
        super().__init__(
            tau=tau_0,
            N_w=N_w,
            N_jitter=N_jitter,
            M_psi=M_psi,
            temperature_max=temperature_max,
            grad_clip=grad_clip,
        )
        self.tau_0 = tau_0
        self.tau_min = tau_min
        self.decay = decay
        self._t = 0

    def _next_tau(self) -> float:
        tau = max(self.tau_min, self.tau_0 * (self.decay ** self._t))
        self._t += 1
        return tau

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        self.tau = self._next_tau()
        return super().compute(system, x)
