"""Differentiable surrogate for Holistic Local EDF analysis.

Based on equations from "Optimized Deadline Assignment and Schedulability
Analysis for Distributed Real-Time Systems with Local EDF Scheduling".

The surrogate is fully differentiable w.r.t. task deadlines using smooth
approximations of ceil, floor, and conditional operations. It is designed
to serve as a cost-function proxy inside a gradient-based optimiser.
"""

import torch
import torch.nn.functional as F

from model.linear_system import LinearSystem


# =========================================================================
# Helpers: build tensor structures from a LinearSystem
# =========================================================================

def _build_system_tensors(system: LinearSystem):
    """Convert a LinearSystem into the tensors needed by the surrogate.

    Returns
    -------
    n : int
        Number of tasks.
    C : (n,) float tensor
        Worst-case execution times.
    T : (n,) float tensor
        Periods (= flow periods).
    J_init : (n,) float tensor
        Initial jitter approximation (predecessor WCET, or 0).
    same_proc : (n, n) bool tensor
        ``same_proc[i, j]`` is True iff tasks i and j share a processor.
    pred_idx : list[int]
        ``pred_idx[i]`` = index of task i's predecessor, or -1.
    last_idx : list[int]
        Indices of the last task in each flow.
    max_d : float
        Maximum deadline value in the system (for sigmoid decoding).
    """
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
# Busy period (exact, non-differentiable — does not depend on deadlines)
# =========================================================================

def _busy_period_tensor(C, T, J, same_proc, max_iters=200, tol=1e-8):
    """Compute the busy period L_i for every task (vectorized).

    Eq (5):  L = ceil(L/T_i)·C_i + Σ_{j≠i} ceil((L+J_j)/T_j)·C_j

    Parameters are (n,) tensors.  Returns (n,) tensor detached from graph.
    """
    n = C.shape[0]
    dtype = C.dtype

    # Start from C_i
    L = C.clone()

    # Expand for broadcasting: (n, 1) vs (1, n)
    for _ in range(max_iters):
        L_prev = L
        # Own contribution: ceil(L_i / T_i) * C_i
        own = torch.ceil(L / T) * C
        # Interference: sum over j≠i on same proc  ceil((L_i + J_j)/T_j) * C_j
        L_exp = L.unsqueeze(1)                      # (n, 1)
        j_mask = same_proc & ~torch.eye(n, dtype=torch.bool)  # (n, n)
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

def _softceil(x, tau=0.5):
    """Smooth ceil:  ceil(x) ≈ x + σ((frac(x) − 0.5)/τ)  where frac ∈ [0,1).

    The sigmoid creates a soft step at the half-integer boundary.
    """
    frac = x - torch.floor(x.detach())  # detach floor → straight-through on frac
    step = torch.sigmoid((frac - 0.5) / tau)
    return x + step


def _softfloor(x, tau=0.5):
    """Smooth floor:  floor(x) ≈ x − σ((frac(x) − 0.5)/τ)."""
    frac = x - torch.floor(x.detach())
    step = torch.sigmoid((frac - 0.5) / tau)
    return x - step


# =========================================================================
# Interference function Wi (Eq 1)
# =========================================================================

def _Wi(w, psi, J_j, T_j, C_j, D_j, tau):
    """Smooth interference from task j during interval w with local deadline psi.

    Eq (1):
        pl = ceil((w + J_j) / T_j)
        pd = 0           if psi < D_j  else  floor((J_j + psi − D_j) / T_j) + 1
        m  = min(pl, pd)
        Wi = m * C_j  if m > 0 else 0

    The condition ``psi < D_j`` is softened with a sigmoid gate.

    All inputs are scalars (0-d tensors).  Returns scalar tensor.
    """
    pl = _softceil((w + J_j) / T_j, tau)
    pl = torch.clamp(pl, min=0.0)

    # Soft gate: ≈1 when psi ≥ D_j, ≈0 when psi < D_j
    gate = torch.sigmoid((psi - D_j) / tau)
    raw_pd = _softfloor((J_j + psi - D_j) / T_j, tau) + 1.0
    pd = gate * raw_pd
    pd = torch.clamp(pd, min=0.0)

    # Smooth minimum
    m = torch.min(pl, pd)
    return m * C_j


# =========================================================================
# Main surrogate forward
# =========================================================================

def edf_surrogate_forward(
    system: LinearSystem,
    s: torch.Tensor,
    tau: float = 0.5,
    N_w: int = 10,
    N_jitter: int = 2,
    M_psi: int = 50,
    temperature_max: float = 0.1,
):
    """Differentiable EDF response-time surrogate.

    Parameters
    ----------
    system : LinearSystem
        The real-time system (tasks, processors, flows).
    s : (n,) tensor
        Sigmoid-encoded deadlines (requires_grad).
    tau : float
        Temperature for soft approximations (ceil, floor, comparisons).
    N_w : int
        Unrolled iterations for the *w_ab* fixed point (Eq 8).
    N_jitter : int
        Outer iterations updating jitter from predecessor response times.
    M_psi : int
        Number of ψ grid points per job *p*.
    temperature_max : float
        Temperature for the soft-max across (p, ψ) when computing WCRT.

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

    # Decode sigmoid values → actual deadlines
    deadlines = s * max_d                                              # (n,)

    # Busy period — exact, detached (deadline-independent)
    L = _busy_period_tensor(C, T, J_init, same_proc)                  # (n,)
    P_per_task = torch.ceil(L / T).clamp(min=1).to(torch.int32)       # (n,)
    P_max = int(P_per_task.max().item())

    if P_max < 1:
        return C.clone().detach().requires_grad_(False), s

    # ---------- outer jitter loop ----------
    J = J_init.clone().to(device)
    r_max = C.clone().detach().to(device)   # placeholder, updated below

    for _ in range(max(N_jitter, 1)):
        # p range: (P_max,)
        p_vals = torch.arange(1, P_max + 1, dtype=dtype, device=device)  # (P,)

        # ψ grid base:  psi_i(p) = (p−1)·T_i + D_i
        # shape (n, P)
        psi_base = (p_vals.unsqueeze(0) - 1.0) * T.unsqueeze(1) + deadlines.unsqueeze(1)

        # fine grid inside [psi_base, psi_base + T_i)  → (n, P, M)
        psi_step = T.unsqueeze(1).unsqueeze(2) / M_psi
        psi_grid = (psi_base.unsqueeze(2)
                    + torch.arange(M_psi, dtype=dtype, device=device) * psi_step)

        # → broadcast psi_grid to (n, n, P, M) for W_i
        psi_brd = psi_grid.unsqueeze(1)                                # (n, 1, P, M)

        # Initial w = p·C_i   → (n, P, M)
        w = p_vals.unsqueeze(0).unsqueeze(2) * C.unsqueeze(1).unsqueeze(2)

        # Validity mask: which (i, p) pairs are within busy period
        p_mask = (p_vals.unsqueeze(0) <= P_per_task.unsqueeze(1).to(device))  # (n, P)
        p_mask_3d = p_mask.unsqueeze(2).to(dtype)                              # (n, P, 1)

        # ---------- inner w_ab fixed-point unrolling ----------
        for __ in range(N_w):
            # pl = softceil((w + J_j) / T_j)     for all j
            # Broadcasting: w (n, P, M), J (n,) → (1, n, 1, 1) vs (n, 1, P, M)
            w_brd = w.unsqueeze(1)                                      # (n, 1, P, M)
            Jj = J.unsqueeze(0).unsqueeze(2).unsqueeze(3)               # (1, n, 1, 1)
            Tj = T.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            Cj = C.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            Dj = deadlines.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            pl = _softceil((w_brd + Jj) / Tj, tau)                     # (n, n, P, M)
            pl = torch.clamp(pl, min=0.0)

            # pd = gate * (softfloor((J_j+psi-D_j)/T_j) + 1)
            gate = torch.sigmoid((psi_brd - Dj) / tau)                  # (n, n, P, M)
            raw_pd = _softfloor((Jj + psi_brd - Dj) / Tj, tau) + 1.0
            pd = gate * raw_pd
            pd = torch.clamp(pd, min=0.0)

            Wi_vals = torch.min(pl, pd) * Cj                           # (n, n, P, M)

            # Zero self-interference, keep only same-processor tasks
            proc_mask = (same_proc.unsqueeze(2).unsqueeze(3).to(dtype)
                         * (1.0 - torch.eye(n, dtype=dtype, device=device)
                            .unsqueeze(2).unsqueeze(3)))
            interference = (Wi_vals * proc_mask).sum(dim=1)             # (n, P, M)

            # w_ab = p·C_i + Σ Wi
            w_new = (p_vals.unsqueeze(0).unsqueeze(2) * C.unsqueeze(1).unsqueeze(2)
                     + interference)

            # Only update valid (i,p) entries
            w = w_new * p_mask_3d + w * (1.0 - p_mask_3d)

        # Response times  r = w − ψ + D_i + J_i
        r = (w - psi_grid
             + deadlines.unsqueeze(1).unsqueeze(2)
             + J.unsqueeze(1).unsqueeze(2))                             # (n, P, M)

        # Mask out invalid (i, p) with large negative for softmax
        r_masked = r * p_mask_3d - 1e9 * (1.0 - p_mask_3d)
        r_flat = r_masked.reshape(n, -1)                                # (n, P*M)

        r_max_new = (torch.logsumexp(r_flat / temperature_max, dim=1)
                     * temperature_max)                                  # (n,)

        # Update jitter from predecessors
        J_new = torch.zeros(n, dtype=dtype, device=device)
        for i in range(n):
            pi = pred_idx[i]
            if pi >= 0:
                J_new[i] = r_max_new[pi].detach()         # detach for stability
            else:
                J_new[i] = J_init[i]

        J = J_new
        r_max = r_max_new

    return r_max, s
