"""Differentiable surrogate for Holistic Local EDF analysis.

Based on equations from "Optimized Deadline Assignment and Schedulability
Analysis for Distributed Real-Time Systems with Local EDF Scheduling".

Provides a fully-differentiable surrogate model of the Holistic Local EDF
analysis, together with GradientFunction implementations that use the
surrogate for fast gradient computation inside the gradient-descent
optimisation framework.
"""

import numpy as np
import torch
import torch.nn.functional as F

from gradient_descent.interfaces import GradientFunction
from model.linear_system import LinearSystem


# =========================================================================
# Tensor builders
# =========================================================================

def _build_system_tensors(system: LinearSystem):
    """Convert a LinearSystem into the tensors needed by the surrogate."""
    tasks = system.tasks
    n = len(tasks)
    dtype = torch.float64
    t2i = {t: i for i, t in enumerate(tasks)}

    C = torch.tensor([t.wcet for t in tasks], dtype=dtype)
    T = torch.tensor([t.period for t in tasks], dtype=dtype)

    J_init = torch.zeros(n, dtype=dtype)
    pred_idx = []
    for i, t in enumerate(tasks):
        if t.predecessors:
            pi = t2i[t.predecessors[0]]
            pred_idx.append(pi)
            J_init[i] = C[pi].clone().detach()
        else:
            pred_idx.append(-1)

    last_idx = [i for i, t in enumerate(tasks) if t.is_last]

    same_proc = torch.zeros(n, n, dtype=torch.bool)
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            same_proc[i, j] = (ti.processor == tj.processor)

    max_d = max(t.deadline for t in tasks) if tasks else 1.0

    return n, C, T, J_init, pred_idx, last_idx, same_proc, max_d


# =========================================================================
# Busy period (exact, non-differentiable — deadline-independent)
# =========================================================================

def _busy_period_tensor(C, T, J, same_proc, max_iters=200, tol=1e-8):
    """Compute busy period L_i for every task (vectorized).  Eq (5)."""
    n = C.shape[0]
    dtype = C.dtype
    L = C.clone()
    for _ in range(max_iters):
        own = torch.ceil(L / T) * C
        L_exp = L.unsqueeze(1)
        j_mask = same_proc & ~torch.eye(n, dtype=torch.bool)
        interference = (torch.ceil((L_exp + J.unsqueeze(0)) / T.unsqueeze(0))
                        * C.unsqueeze(0) * j_mask.to(dtype)).sum(dim=1)
        L_new = own + interference
        if torch.allclose(L_new, L, atol=tol):
            return L_new.detach()
        L = L_new
    return L.detach()


# =========================================================================
# Smooth arithmetic
# =========================================================================

def _softceil(x, tau):
    """Smooth ceil via sigmoid step at half-integer boundary."""
    frac = x - torch.floor(x.detach())
    step = torch.sigmoid((frac - 0.5) / max(tau, 1e-9))
    return x + step


def _softfloor(x, tau):
    """Smooth floor via sigmoid step at half-integer boundary."""
    frac = x - torch.floor(x.detach())
    step = torch.sigmoid((frac - 0.5) / max(tau, 1e-9))
    return x - step


# =========================================================================
# Surrogate forward pass
# =========================================================================

def surrogate_edf_forward(
    system: LinearSystem,
    s: torch.Tensor,
    tau: float = 0.05,
    N_w: int = 10,
    N_jitter: int = 2,
    M_psi: int = 50,
    temperature_max: float = 0.1,
):
    """Differentiable EDF response-time surrogate.

    Parameters
    ----------
    system : LinearSystem
    s : (n,) tensor
        Sigmoid-encoded deadlines (requires_grad=True).
    tau : float
        Temperature for soft ceil / floor / comparisons.
    N_w : int
        Unrolled iterations for the *w_ab* fixed point (Eq 8).
    N_jitter : int
        Outer passes updating jitter from predecessor response times.
    M_psi : int
        Number of psi grid points per job *p*.
    temperature_max : float
        Temperature for the soft-max over (p, psi).

    Returns
    -------
    r_max : (n,) tensor
        Approximate worst-case response times.
    s : (n,) tensor
        The input leaf tensor (for autograd).
    """
    (n, C, T, J_init, pred_idx, last_idx,
     same_proc, max_d) = _build_system_tensors(system)

    dtype = C.dtype
    device = s.device

    deadlines = s * max_d                                              # (n,)

    L = _busy_period_tensor(C, T, J_init, same_proc)                  # (n,)
    P_per_task = torch.ceil(L / T).clamp(min=1).to(torch.int32)       # (n,)
    P_max = int(P_per_task.max().item())

    if P_max < 1:
        return C.clone().detach().requires_grad_(False), s

    # ---------- outer jitter loop ----------
    J = J_init.clone().to(device)

    for _ in range(max(N_jitter, 1)):
        p_vals = torch.arange(1, P_max + 1, dtype=dtype, device=device)  # (P,)

        psi_base = (p_vals.unsqueeze(0) - 1.0) * T.unsqueeze(1) + deadlines.unsqueeze(1)
        psi_step = T.unsqueeze(1).unsqueeze(2) / M_psi
        psi_grid = (psi_base.unsqueeze(2)
                    + torch.arange(M_psi, dtype=dtype, device=device) * psi_step)

        psi_brd = psi_grid.unsqueeze(1)                                # (n, 1, P, M)

        w = p_vals.unsqueeze(0).unsqueeze(2) * C.unsqueeze(1).unsqueeze(2)
        p_mask = (p_vals.unsqueeze(0) <= P_per_task.unsqueeze(1).to(device))
        p_mask_3d = p_mask.unsqueeze(2).to(dtype)

        # ---------- inner w_ab fixed-point unrolling ----------
        for __ in range(N_w):
            w_brd = w.unsqueeze(1)                                      # (n, 1, P, M)
            Jj = J.unsqueeze(0).unsqueeze(2).unsqueeze(3)               # (1, n, 1, 1)
            Tj = T.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            Cj = C.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            Dj = deadlines.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            pl = _softceil((w_brd + Jj) / Tj, tau)
            pl = torch.clamp(pl, min=0.0)

            gate = torch.sigmoid((psi_brd - Dj) / tau)
            raw_pd = _softfloor((Jj + psi_brd - Dj) / Tj, tau) + 1.0
            pd = gate * raw_pd
            pd = torch.clamp(pd, min=0.0)

            Wi_vals = torch.min(pl, pd) * Cj                           # (n, n, P, M)

            proc_mask = (same_proc.unsqueeze(2).unsqueeze(3).to(dtype)
                         * (1.0 - torch.eye(n, dtype=dtype, device=device)
                            .unsqueeze(2).unsqueeze(3)))
            interference = (Wi_vals * proc_mask).sum(dim=1)             # (n, P, M)

            w_new = (p_vals.unsqueeze(0).unsqueeze(2) * C.unsqueeze(1).unsqueeze(2)
                     + interference)
            w = w_new * p_mask_3d + w * (1.0 - p_mask_3d)

        r = (w - psi_grid
             + deadlines.unsqueeze(1).unsqueeze(2)
             + J.unsqueeze(1).unsqueeze(2))                             # (n, P, M)

        r_masked = r * p_mask_3d - 1e9 * (1.0 - p_mask_3d)
        r_flat = r_masked.reshape(n, -1)

        r_max_new = (torch.logsumexp(r_flat / temperature_max, dim=1)
                     * temperature_max)                                  # (n,)

        # Update jitter from predecessors
        J_new = torch.zeros(n, dtype=dtype, device=device)
        for i in range(n):
            pi = pred_idx[i]
            J_new[i] = r_max_new[pi].detach() if pi >= 0 else J_init[i]
        J = J_new

    return r_max_new, s


# =========================================================================
# GradientFunction implementations
# =========================================================================

class SurrogateEDFGradient(GradientFunction):
    """Gradient via the differentiable EDF surrogate + autograd.

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
        tau: float = 0.05,
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

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        tensors = _build_system_tensors(system)
        last_idx = tensors[5]
        n = len(system.tasks)
        dtype = torch.float64

        s = torch.tensor(x, dtype=dtype, requires_grad=True)

        try:
            r, s_leaf = surrogate_edf_forward(
                system, s,
                tau=self.tau,
                N_w=self.N_w,
                N_jitter=self.N_jitter,
                M_psi=self.M_psi,
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


class SurrogateEDFAnnealedGradient(SurrogateEDFGradient):
    """SurrogateEDFGradient with an annealing schedule for tau.

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
