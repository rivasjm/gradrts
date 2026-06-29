"""Differentiable V1 surrogate — Phases 1, 2, and 3.

Phase 1 – Unrolled V1
    N iterations of the V1 update, fully differentiable.
    The *p* loop is relaxed via softplus (smooth ceil).

Phase 2 – Annealing
    Temperature τ decays during optimisation; use the same forward
    calls with a schedule τ(iter).

Phase 3 – Implicit differentiation
    Solves r = F(r, θ) to convergence; PyTorch's autograd handles the
    implicit gradient through the fixed-point computation graph.
"""

import torch
import torch.nn.functional as F

from model.linear_system import LinearSystem


# ======================================================================
# Helper: build task indices
# ======================================================================

def _build_indices(system: LinearSystem):
    tasks = system.tasks
    n = len(tasks)
    t2i = {t: i for i, t in enumerate(tasks)}

    pred_idx = [-1] * n
    for i, t in enumerate(tasks):
        p = t.predecessors
        pred_idx[i] = t2i[p[0]] if p else -1

    last_idx = [i for i, t in enumerate(tasks) if t.is_last]

    same_proc = torch.zeros(n, n, dtype=torch.bool)
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            same_proc[i, j] = (ti.processor == tj.processor)

    return pred_idx, last_idx, same_proc


# ======================================================================
# V1 soft-priority step  (used by both unrolled and implicit)
# ======================================================================

def _v1_soft_step(r, C, T, pred_idx, same_proc, s, tau, P_max=20, beta=10.0):
    """One V1 iteration — fully differentiable with soft p-loop.

    r, C, T, s are (n,) tensors.
    Returns r_new (n,).
    """
    n = len(C)
    dtype = C.dtype

    U = C / T

    # Soft hp: hp[i,j] = P(j ∈ hp(i)) = σ((s_j - s_i) / τ)
    hp_soft = torch.sigmoid((s.unsqueeze(1) - s.unsqueeze(0)) / tau)
    mask = same_proc.clone()
    mask.fill_diagonal_(False)
    hp_soft = hp_soft * mask.to(dtype)

    # D_i = 1 − Σ_j u_j · hp[i,j]
    D = 1.0 - (U.unsqueeze(0) * hp_soft).sum(dim=1)

    # Jitter: J_i = r[pred(i)] or 0
    J = torch.zeros(n, dtype=dtype)
    for i in range(n):
        pi = pred_idx[i]
        if pi >= 0:
            J[i] = r[pi]

    # JU_i = Σ_j J_j · u_j · hp[i,j]
    JU = (J.unsqueeze(0) * U.unsqueeze(0) * hp_soft).sum(dim=1)

    # --- Soft p-loop ---
    # Optimal p:  p* = max(1, ceil(JU / (D·T - C)))
    # Use softplus as smooth ceil:  p_soft = 1 + softplus(ξ, β)  where ξ = JU/(D·T-C) - 1
    denom_p = D * T - C          # D·T - C; if ≤ 0 → unschedulable
    safe = denom_p > 1e-9

    xi = torch.zeros(n, dtype=dtype)
    xi[safe] = JU[safe] / denom_p[safe] - 1.0
    xi[~safe] = float('inf')

    p_soft = 1.0 + F.softplus(xi, beta=beta) / beta

    # Clamp to [1, P_max]
    p_soft = p_soft.clamp(1.0, float(P_max))

    # r = (p*C + JU) / D  -  (p-1)*T  +  J
    w_p = (p_soft * C + JU) / D
    r_new = w_p - (p_soft - 1.0) * T + J

    # Handle D ≤ 0 (processor overloaded)
    r_new[D <= 0] = float('inf')

    return r_new


# ======================================================================
# Phase 1 – Unrolled V1
# ======================================================================

def v1_unrolled_forward(
    system: LinearSystem,
    tau: float = 0.5,
    N: int = 10,
    P_max: int = 20,
    s_init: torch.Tensor = None,
):
    """Unrolled V1: N differentiable iterations.

    Returns (r, s) where s is the leaf tensor for autograd.
    """
    tasks = system.tasks
    n = len(tasks)
    dtype = torch.float64

    pred_idx, _last_idx, same_proc = _build_indices(system)

    C = torch.tensor([t.wcet for t in tasks], dtype=dtype)
    T = torch.tensor([t.period for t in tasks], dtype=dtype)

    if s_init is not None:
        s = s_init.clone().detach().to(dtype).requires_grad_(True)
    else:
        s = torch.tensor([float(t.priority) for t in tasks],
                         dtype=dtype, requires_grad=True)

    r = C.clone()
    for i in range(n):
        pi = pred_idx[i]
        if pi >= 0:
            r[i] = r[i] + r[pi]

    for _ in range(N):
        r = _v1_soft_step(r, C, T, pred_idx, same_proc, s, tau, P_max)

    return r, s


# ======================================================================
# Phase 3 – Implicit differentiation
# ======================================================================

def v1_implicit_forward(
    system: LinearSystem,
    tau: float = 0.5,
    max_iters: int = 200,
    tol: float = 1e-8,
    P_max: int = 20,
    s_init: torch.Tensor = None,
):
    """Solve V1 to convergence; gradient via implicit differentiation.

    Returns (r*, s) where s is the leaf tensor.
    """
    tasks = system.tasks
    n = len(tasks)
    dtype = torch.float64

    pred_idx, _last_idx, same_proc = _build_indices(system)

    C = torch.tensor([t.wcet for t in tasks], dtype=dtype)
    T = torch.tensor([t.period for t in tasks], dtype=dtype)

    if s_init is not None:
        s = s_init.clone().detach().to(dtype).requires_grad_(True)
    else:
        s = torch.tensor([float(t.priority) for t in tasks],
                         dtype=dtype, requires_grad=True)

    r = C.clone()
    for i in range(n):
        pi = pred_idx[i]
        if pi >= 0:
            r[i] = r[i] + r[pi]

    for _ in range(max_iters):
        r_new = _v1_soft_step(r, C, T, pred_idx, same_proc, s, tau, P_max)
        diff = (r_new - r).abs().max().item()
        r = r_new
        if diff < tol:
            break

    return r, s


# ======================================================================
# Phase 2 – Annealing schedule
# ======================================================================

def exponential_decay(tau_0: float = 2.0, tau_min: float = 0.05, decay: float = 0.95):
    """Return a schedule callable:  τ(t) = max(tau_min, tau_0 * decay^t)."""
    def schedule(t: int) -> float:
        return max(tau_min, tau_0 * (decay ** t))
    return schedule
