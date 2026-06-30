"""Differentiable EDF surrogate v2 — uses the actual psi set (interfering
deadlines) instead of a uniform grid.

This brings the surrogate much closer to the real HolisticLocalEDFAnalysis
by evaluating w_ab at the same critical psi points (own + interfering tasks'
absolute deadlines).  Soft sigmoid gates preserve differentiability.
"""

import torch
import torch.nn.functional as F

from model.linear_system import LinearSystem


# =========================================================================
# Helpers: build tensor structures from a LinearSystem
# =========================================================================

def _build_system_tensors(system: LinearSystem):
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
# Busy period (exact, non-differentiable)
# =========================================================================

def _busy_period_tensor(C, T, J, same_proc, max_iters=200, tol=1e-8):
    n = C.shape[0]
    dtype = C.dtype
    L = C.clone()
    for _ in range(max_iters):
        L_prev = L
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

def _softceil(x, tau=0.1):
    frac = x - torch.floor(x.detach())
    step = torch.sigmoid((frac - 0.5) / max(tau, 1e-9))
    return x + step


def _softfloor(x, tau=0.1):
    frac = x - torch.floor(x.detach())
    step = torch.sigmoid((frac - 0.5) / max(tau, 1e-9))
    return x - step


# =========================================================================
# Build psi candidates for a single task i
# =========================================================================

def _build_psi_candidates(i, deadlines, C, T, J, L, same_proc, device, dtype):
    """Return (psi_vals, own_idx_mask) for task i — all tensor ops, no .item().

    psi_vals : (K,) tensor of all candidate psi values sorted.
    own_idx_mask : (K,) bool tensor — True for own deadlines.
    """
    n = C.shape[0]
    Ti = T[i]
    Di = deadlines[i]
    Li = L[i]

    Pi = max(1, int(torch.ceil(Li / Ti).item()))
    p_vals = torch.arange(1, Pi + 1, dtype=dtype, device=device)
    own = (p_vals - 1) * Ti + Di                  # (P_i,) — stays in graph

    # Interfering tasks' absolute deadlines within [D_i, L_i + D_i]
    psi_low = Di
    psi_high = Di + Li

    interfering_parts = []
    for j in range(n):
        if j == i or not same_proc[i, j]:
            continue
        Q_ij = max(1, int(torch.ceil((Li + J[j]) / T[j]).item()))
        q_vals = torch.arange(1, Q_ij + 1, dtype=dtype, device=device)
        # psi_j(q) = (q-1)*T_j - J_j + D_j   (J_j is detached, T_j is const, D_j stays in graph)
        psi_jq = (q_vals - 1) * T[j] - J[j] + deadlines[j]   # (Q_ij,)
        # Keep only those in [D_i, L_i + D_i]
        in_range = (psi_jq >= psi_low - 1e-6) & (psi_jq <= psi_high + 1e-6)
        if in_range.any():
            interfering_parts.append(psi_jq[in_range])

    if interfering_parts:
        interfering = torch.cat(interfering_parts)
    else:
        interfering = torch.empty(0, dtype=dtype, device=device)

    all_psi = torch.cat([own, interfering])
    all_psi, sort_idx = torch.sort(all_psi)

    # Mark which are own deadlines (compare with original own values)
    own_mask = torch.zeros(all_psi.shape[0], dtype=torch.bool, device=device)
    for p in range(Pi):
        matches = torch.isclose(all_psi, own[p])
        own_mask = own_mask | matches

    return all_psi, own_mask


# =========================================================================
# Main surrogate forward (v2 — real psi set)
# =========================================================================

def edf_surrogate_forward_v2(
    system: LinearSystem,
    s: torch.Tensor,
    tau: float = 0.1,
    N_w: int = 10,
    N_jitter: int = 2,
    temperature_max: float = 0.1,
):
    (n, C, T, J_init, pred_idx, last_idx,
     same_proc, max_d) = _build_system_tensors(system)

    dtype = C.dtype
    device = s.device

    deadlines = s * max_d                                              # (n,)

    # Busy period — exact, detached
    L = _busy_period_tensor(C, T, J_init, same_proc)                  # (n,)

    # ---------- outer jitter loop ----------
    J = J_init.clone().to(device)

    for _ in range(max(N_jitter, 1)):
        r_max_list = []

        for i in range(n):
            Ti = T[i]
            Ti_val = Ti.item()

            # Build psi candidates for task i
            psi_vals, own_mask = _build_psi_candidates(
                i, deadlines, C, T, J, L, same_proc, device, dtype)
            K = psi_vals.shape[0]

            if K == 0:
                r_max_list.append(C[i].clone().detach().unsqueeze(0))
                continue

            Pi = max(1, int(torch.ceil(L[i] / Ti).item()))
            P_max_task = Pi

            # For each p, lower[p] = (p-1)*T_i + D_i, upper[p] = p*T_i + D_i
            p_vals = torch.arange(1, P_max_task + 1, dtype=dtype, device=device)  # (P,)
            lower = (p_vals - 1.0) * Ti + deadlines[i]                             # (P,)
            upper = p_vals * Ti + deadlines[i]                                      # (P,)

            # Soft gate: which psi_vals fall in [lower[p], upper[p])
            # gate[p, k] ≈ 1 if psi_vals[k] ∈ [lower[p], upper[p])
            psi_exp = psi_vals.unsqueeze(0)       # (1, K)
            lower_exp = lower.unsqueeze(1)         # (P, 1)
            upper_exp = upper.unsqueeze(1)         # (P, 1)

            gate_in = (torch.sigmoid((psi_exp - lower_exp) / tau)
                       * torch.sigmoid((upper_exp - psi_exp) / tau))  # (P, K)

            # Own deadlines always get weight 1.0 in their own interval
            own_exp = own_mask.unsqueeze(0).to(dtype)                  # (1, K)
            # For own deadline at position k belonging to interval p:
            # it's at psi = lower[p], so the gate should be 1.0
            # Already handled by sigmoid gates (psi - lower = 0 → sigmoid(0)=0.5, 
            # but product with upper gate gives ~0.25). We boost it:
            gate_in = gate_in + own_exp * 1.0
            gate_in = torch.clamp(gate_in, max=1.0)

            # Initial w = p*C_i for each (p, k)
            w = (p_vals.unsqueeze(1) * C[i]).expand(P_max_task, K)     # (P, K)

            # ---------- inner w_ab fixed-point unrolling ----------
            for __ in range(N_w):
                # Compute Wi(j, w, psi) for all j on same processor
                w_exp = w.unsqueeze(0)                                 # (1, P, K)
                psi_exp2 = psi_vals.unsqueeze(0).unsqueeze(0)          # (1, 1, K)

                # Broadcast task j params: (n, 1, 1)
                Jj = J.unsqueeze(1).unsqueeze(2)
                Tj = T.unsqueeze(1).unsqueeze(2)
                Cj = C.unsqueeze(1).unsqueeze(2)
                Dj = deadlines.unsqueeze(1).unsqueeze(2)

                # Expand w and psi for j-broadcasting: (n, P, K)
                w_brd = w_exp.expand(n, -1, -1)
                psi_brd = psi_exp2.expand(n, -1, -1)

                pl = _softceil((w_brd + Jj) / Tj, tau)
                pl = torch.clamp(pl, min=0.0)

                gate_j = torch.sigmoid((psi_brd - Dj) / tau)
                raw_pd = _softfloor((Jj + psi_brd - Dj) / Tj, tau) + 1.0
                pd = gate_j * raw_pd
                pd = torch.clamp(pd, min=0.0)

                Wi_vals = torch.min(pl, pd) * Cj                       # (n, P, K)

                # Mask: j != i, same processor as task i
                # same_proc[i] is (n,) bool; reshape to (n,1,1)
                mask_j = same_proc[i].to(dtype).unsqueeze(1).unsqueeze(2)  # (n, 1, 1)
                mask_j[i] = 0.0  # exclude self
                interference = (Wi_vals * mask_j).sum(dim=0)             # (P, K)

                # w_ab = p*C_i + interference
                w_new = (p_vals.unsqueeze(1) * C[i]) + interference

                w = w_new

            # Response times: r = w - psi + D_i + J_i
            r = (w - psi_vals.unsqueeze(0)
                 + deadlines[i]
                 + J[i])                                                # (P, K)

            # Weight by gate for soft max
            r_weighted = r * gate_in - 1e9 * (1.0 - gate_in)

            # Soft max over (p, k)
            r_flat = r_weighted.reshape(-1)
            r_max_i = (torch.logsumexp(r_flat / temperature_max, dim=0)
                       * temperature_max)
            r_max_list.append(r_max_i.unsqueeze(0))

        r_max_new = torch.cat(r_max_list)                               # (n,)

        # Update jitter from predecessors
        J_new = torch.zeros(n, dtype=dtype, device=device)
        for i in range(n):
            pi = pred_idx[i]
            if pi >= 0:
                J_new[i] = r_max_new[pi].detach()
            else:
                J_new[i] = J_init[i]

        J = J_new

    return r_max_new, s
