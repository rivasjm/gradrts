"""Differentiable V3 analysis for PyTorch autograd.

Two variants:

1. **WCET gradient**  (``v3_forward_torch``)
   Returns (r, C_leaf).  Call ``r.mean().backward()`` then read ``C_leaf.grad``.

2. **Priority-score gradient**  (``v3_soft_priority``)
   Replaces the hard *hp* set with sigmoid-softened membership so that
   the whole pipeline is differentiable w.r.t. per-task priority scores.

   Usage::

       r, s_leaf = v3_soft_priority(C, T, pred, same_proc_mask, tau=0.1)
       avg_wcrt = r[last].mean()
       avg_wcrt.backward()
       ds = s_leaf.grad   # d(avg_wcrt) / d(score_i)
"""

import torch
import torch.nn.functional as F

from model.linear_system import LinearSystem, Task
from workspace.holistic_linearization.linearized_fp import _higher_priority


# ======================================================================
# WCET gradient
# ======================================================================

def v3_forward_torch(system: LinearSystem):
    tasks = system.tasks
    n = len(tasks)
    t2i = {t: i for i, t in enumerate(tasks)}

    pred_idx = [-1] * n
    for i, t in enumerate(tasks):
        p = t.predecessors
        if p:
            pred_idx[i] = t2i[p[0]]

    C = torch.tensor([t.wcet for t in tasks], dtype=torch.float64, requires_grad=True)
    T = torch.tensor([t.period for t in tasks], dtype=torch.float64)
    U = C / T

    D = torch.ones(n, dtype=torch.float64)
    for i, t in enumerate(tasks):
        hp = _higher_priority(t)
        if hp:
            D[i] = 1.0 - U[[t2i[h] for h in hp]].sum()

    b = C / D
    M = torch.zeros(n, n, dtype=torch.float64)
    for i, t in enumerate(tasks):
        di = D[i]
        pi = pred_idx[i]
        if pi >= 0:
            M[i, pi] = 1.0
        for h in _higher_priority(t):
            ph = pred_idx[t2i[h]]
            if ph >= 0:
                M[i, ph] = M[i, ph] + U[t2i[h]] / di

    I = torch.eye(n, dtype=torch.float64)
    r = torch.linalg.solve(I - M, b)
    return r, C


# ======================================================================
# Priority-score gradient (soft hp sets)
# ======================================================================

def v3_soft_priority(
    C: torch.Tensor,
    T: torch.Tensor,
    pred_idx: list[int],
    same_proc: torch.Tensor,
    tau: float = 0.5,
    s_init: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable V3 with softmax-relaxed priority ordering.

    Parameters
    ----------
    C : (n,) Tensor
        WCETs (leaf with ``requires_grad=True``).
    T : (n,) Tensor
        Periods.
    pred_idx : list[int]
        ``pred_idx[i]`` = index of predecessor task (or -1).
    same_proc : (n, n) Tensor of bool
        ``same_proc[i, j] = True`` if tasks i and j are on the same
        processor.
    tau : float
        Temperature for the sigmoid.  Lower = sharper (closer to
        hard priority).  Default 0.5.
    s_init : (n,) Tensor, optional
        Initial priority scores.  Default: all zeros.

    Returns
    -------
    r : (n,) Tensor
        Response-time vector.
    s : (n,) Tensor (leaf, requires_grad=True)
        The learned priority scores.  Read ``s.grad`` for the
        gradient of ``r.mean()`` w.r.t. scores.
    """
    n = len(C)
    dtype = C.dtype

    # --- learnable scores ---
    if s_init is not None:
        s = s_init.clone().detach().to(dtype).requires_grad_(True)
    else:
        s = torch.zeros(n, dtype=dtype, requires_grad=True)

    U = C / T  # (n,)

    # --- soft hp mask:  hp_soft[i, j] ≈ 1 if j higher prio than i ---
    # score_diff[i, j] = s_j - s_i   (positive → j has higher prio than i)
    score_diff = s.unsqueeze(0) - s.unsqueeze(1)  # (n, n): diff[i,j] = s_i - s_j? No...
    # Let's compute: diff where diff[i,j] > 0 means j has higher priority.
    # hp_soft[i, j] = sigmoid((s_j - s_i) / tau)
    hp_soft = torch.sigmoid((s.unsqueeze(1) - s.unsqueeze(0)) / tau)  # (n, n): hp_soft[i,j] = σ((s_j - s_i)/τ)

    # Mask out self and different-processor tasks
    mask = same_proc.clone()
    mask.fill_diagonal_(False)
    hp_soft = hp_soft * mask.to(dtype)  # zero out invalid pairs

    # --- D_i = 1 - sum_{j} u_j * hp_soft[i,j] ---
    D = 1.0 - (U.unsqueeze(0) * hp_soft).sum(dim=1)  # (n,)

    # --- b_i = C_i / D_i ---
    b = C / D

    # --- M matrix ---
    # M[i, k] = (jitter from own predecessor) + sum_j (u_j/D_i) * hp_soft[i,j] * [k == pred(j)]
    M = torch.zeros(n, n, dtype=dtype)

    # own-predecessor jitter (always hard — priority-independent)
    for i in range(n):
        pi = pred_idx[i]
        if pi >= 0:
            M[i, pi] = 1.0

    # interference jitter
    # For each target k, contribution = sum_j (u_j / D_i) * hp_soft[i,j] * [pred(j)==k]
    # Precompute a sparse contrib matrix: for each j with pred[j] = k, contrib_{j,k} = u_j
    # Then M[i, k] = sum_j contrib_{j,k} * hp_soft[i,j] / D_i
    contrib = torch.zeros(n, n, dtype=dtype)  # contrib[j, k] = u_j if pred[j] == k else 0
    for j in range(n):
        pj = pred_idx[j]
        if pj >= 0:
            contrib[j, pj] = U[j]

    # M_extra[i, k] = sum_j contrib[j,k] * hp_soft[i,j] / D_i
    # = hp_soft[i, :] @ contrib[:, k] / D_i
    # M_extra = hp_soft @ contrib  then divide each row by D_i
    M_extra = hp_soft @ contrib  # (n, n): M_extra[i, k] = sum_j contrib[j,k] * hp_soft[i,j]
    M_extra = M_extra / D.unsqueeze(1)  # divide row i by D_i

    M = M + M_extra

    # --- solve ---
    I = torch.eye(n, dtype=dtype)
    r = torch.linalg.solve(I - M, b)

    return r, s
