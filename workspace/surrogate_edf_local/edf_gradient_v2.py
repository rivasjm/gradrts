"""GradientFunction for the v2 surrogate (real psi set)."""

import numpy as np
import torch

from gradient_descent.interfaces import GradientFunction
from model.linear_system import LinearSystem
from workspace.surrogate_edf_local.differentiable_edf_v2 import (
    edf_surrogate_forward_v2,
    _build_system_tensors,
)


class SurrogateEDFGradientV2(GradientFunction):
    def __init__(
        self,
        tau: float = 0.1,
        N_w: int = 10,
        N_jitter: int = 2,
        temperature_max: float = 0.1,
        grad_clip: float = 1.0,
    ):
        self.tau = tau
        self.N_w = N_w
        self.N_jitter = N_jitter
        self.temperature_max = temperature_max
        self.grad_clip = grad_clip

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        tensors = _build_system_tensors(system)
        last_idx = tensors[5]
        n = len(system.tasks)
        dtype = torch.float64

        s = torch.tensor(x, dtype=dtype, requires_grad=True)

        try:
            r, s_leaf = edf_surrogate_forward_v2(
                system, s,
                tau=self.tau,
                N_w=self.N_w,
                N_jitter=self.N_jitter,
                temperature_max=self.temperature_max,
            )
        except Exception:
            return [0.0] * n

        if not last_idx:
            return [0.0] * n

        r_last = r[last_idx]
        max_d = max(t.deadline for t in system.tasks) if system.tasks else 1.0
        D_last = s_leaf[last_idx] * max_d
        slack = (r_last - D_last) / D_last.clamp(min=1e-9)
        cost = torch.logsumexp(slack / self.temperature_max, dim=0) * self.temperature_max

        cost.backward()

        g = s_leaf.grad
        if g is None:
            return [0.0] * n

        grad = g.detach().numpy()
        if np.any(~np.isfinite(grad)):
            return [0.0] * n

        if self.grad_clip is not None:
            gn = np.linalg.norm(grad)
            if gn > self.grad_clip:
                grad = grad * (self.grad_clip / gn)

        return grad.tolist()
