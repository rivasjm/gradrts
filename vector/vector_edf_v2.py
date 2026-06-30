"""Vectorized Holistic Local EDF analysis — v2 (exact psi set).

Evaluates w_ab at the exact same absolute-deadline points used by the
scalar HolisticLocalEDFAnalysis, yielding WCRTs that match within
numerical tolerance (max diff < 1e-4).

Architecture: tasks are processed sequentially (enabling intra-iteration
jitter propagation), but within each task the analysis is vectorised
over all deadline scenarios simultaneously.
"""

import numpy as np

from model.analysis_function import init_wcrt
from gradient_descent.gradient_function import AvgSeparationDelta, gradient_inputs_from_deltas, gradient_from_costs
from gradient_descent.interfaces import GradientFunction
from model.linear_system import LinearSystem


# =========================================================================
# Cache
# =========================================================================

class ResultsCache:
    def __init__(self):
        self.data = dict()

    @staticmethod
    def _key(deadlines):
        return deadlines.tobytes()

    def insert(self, deadlines, results):
        key = self._key(deadlines)
        if key not in self.data:
            self.data[key] = results

    def get(self, deadlines):
        return self.data.get(key := self._key(deadlines))

    def has_results(self, deadlines):
        return self._key(deadlines) in self.data

    def reset(self):
        self.data.clear()


# =========================================================================
# Tensor builders
# =========================================================================

def _get_vectors(system: LinearSystem):
    tasks = system.tasks
    n = len(tasks)
    dtype = np.float32
    t2i = {t: i for i, t in enumerate(tasks)}

    C = np.array([t.wcet for t in tasks], dtype=dtype)
    T = np.array([t.period for t in tasks], dtype=dtype)
    D_task = np.array([t.deadline for t in tasks], dtype=dtype)
    D_flow = np.array([t.flow.deadline for t in tasks], dtype=dtype)
    J_init = np.zeros(n, dtype=dtype)

    pred_idx = np.full(n, -1, dtype=np.int32)
    for i, t in enumerate(tasks):
        if t.predecessors:
            pi = t2i[t.predecessors[0]]
            pred_idx[i] = pi

    # Initial jitter = predecessor's cumulative WCET (matching init_wcrt)
    wcrt_init = C.copy()
    for i in range(n):
        pi = pred_idx[i]
        if pi >= 0:
            wcrt_init[i] += wcrt_init[pi]
            J_init[i] = wcrt_init[pi]

    same_proc = np.zeros((n, n), dtype=np.bool_)
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            same_proc[i, j] = (ti.processor == tj.processor)

    return C, T, D_task, D_flow, J_init, pred_idx, same_proc


def _busy_period(C, T, J, same_proc, max_iters=200, tol=1e-6):
    n = len(C)
    L = C.copy()
    j_mask = same_proc & ~np.eye(n, dtype=bool)
    for _ in range(max_iters):
        own = np.ceil(L / T) * C
        L_exp = L.reshape(1, n)
        interference = np.sum(
            np.ceil((L_exp + J.reshape(1, n)) / T.reshape(1, n))
            * C.reshape(1, n) * j_mask, axis=1
        )
        L_new = own + interference
        if np.allclose(L_new, L, atol=tol):
            return L_new
        L = L_new
    return L


# =========================================================================
# w_ab fixed-point helper — vectorised over (scenarios, psi_points)
# =========================================================================

def _compute_w(
    w_init, psi_vals, p, Ci, J, D, C, T, same_proc_i,
    max_iters=200, tol=1e-4,
):
    """Compute w_ab for multiple psi values, vectorised over scenarios.

    Parameters
    ----------
    w_init : (s, K) float32
        Initial workload values.
    psi_vals : (s, K) float32
        Deadline values under analysis.
    p : int
        Job index (1-based).
    Ci : float
        WCET of the task under analysis.
    J : (s, n) float32
        Current jitter per scenario per task.
    D : (s, n) float32
        Per-task deadlines per scenario.
    C, T : (n,) float32
        WCETs and periods.
    same_proc_i : (n,) bool
        Processor mask for task i.

    Returns
    -------
    w : (s, K) float32
    """
    s, K = w_init.shape
    n = len(C)
    w = w_init.copy()

    # Pre-build broadcast tensors
    C_brd = C[None, None, :]                                            # (1, 1, n)
    T_brd = T[None, None, :]
    mask_j = same_proc_i.astype(np.float32)                             # (n,)
    mask_j_brd = mask_j[None, None, :]                                  # (1, 1, n)

    for _ in range(max_iters):
        w_prev = w.copy()

        w_brd = w[:, :, np.newaxis]                                     # (s, K, 1)
        Jj = J[:, np.newaxis, :]                                        # (s, 1, n)
        Dj = D[:, np.newaxis, :]                                        # (s, 1, n)
        psi_brd = psi_vals[:, :, np.newaxis]                            # (s, K, 1)

        pl = np.maximum(0.0, np.ceil((w_brd + Jj) / T_brd))
        cond = (psi_brd >= Dj).astype(np.float32)
        raw_pd = np.maximum(0.0, np.floor((Jj + psi_brd - Dj) / T_brd) + 1.0)
        pd = cond * raw_pd
        Wi = np.minimum(pl, pd) * C_brd                                 # (s, K, n)

        interference = np.sum(Wi * mask_j_brd, axis=2)                  # (s, K)
        w_new = p * Ci + interference
        w = w_new.astype(np.float32)
        if np.allclose(w, w_prev, atol=tol):
            break

    return w


# =========================================================================
# Vectorized Holistic EDF Analysis — v2
# =========================================================================

class VectorHolisticEDFAnalysisV2:
    """Exact vectorised EDF analysis using the real psi set."""

    def __init__(self, limit_factor=10, cache=None):
        self.limit_factor = limit_factor
        self.cache = cache if cache is not None else ResultsCache()
        self._scenarios_response_times = None
        self._full_response_times = None

    def clear_results(self):
        self._scenarios_response_times = None
        self._full_response_times = None

    @property
    def scenarios_response_times(self):
        return self._scenarios_response_times

    @property
    def full_response_times(self):
        return self._full_response_times

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def _analysis(self, D_scenarios, D_flow, C, T, J_init, pred_idx, same_proc):
        s_total = D_scenarios.shape[0]
        n = C.shape[0]
        limit = self.limit_factor
        cache = self.cache

        # Remove already-cached scenarios
        keep = np.ones(s_total, dtype=bool)
        for k in range(s_total):
            if cache.has_results(D_scenarios[k]):
                keep[k] = False
        D_work = D_scenarios[keep]
        s = D_work.shape[0]
        if s == 0:
            return _rebuild_from_cache(D_scenarios, cache)

        # Initial jitter and WCRT
        J = np.tile(J_init, (s, 1)).astype(np.float32)
        D = D_work.astype(np.float32)
        D_limit = D_flow.reshape(1, n) * limit

        WCRT = np.tile(C.reshape(1, n), (s, 1))
        for i in range(n):
            pi = pred_idx[i]
            if pi >= 0:
                WCRT[:, i] += WCRT[:, pi]

        # Topological order for intra-iteration jitter propagation
        order = _topological_order(n, pred_idx)

        # --- Outer convergence loop ---
        for outer in range(50):
            WCRT_prev = WCRT.copy()

            # Recompute busy periods with max jitter across all scenarios (conservative)
            J_ref = np.max(J, axis=0) if s > 0 else J_init
            L = _busy_period(C, T, J_ref, same_proc)
            P_vals = np.maximum(1, np.ceil(L / T).astype(np.int32))

            # Recompute Q_max with current jitter
            Q_max_per_j = np.zeros(n, dtype=np.int32)
            for j in range(n):
                qs = []
                for i in range(n):
                    if same_proc[i, j] and i != j:
                        qs.append(int(np.ceil((L[i] + J_ref[j]) / T[j])))
                Q_max_per_j[j] = max(qs) if qs else 0

            for i in order:
                Pi = int(P_vals[i])
                Ti = T[i]
                Ci = C[i]
                same_proc_i = same_proc[i].copy()
                same_proc_i[i] = False

                interfering = [j for j in range(n)
                               if same_proc[i, j] and j != i and Q_max_per_j[j] > 0]

                wcrt_i = np.full(s, 0.0, dtype=np.float32)

                for p_idx in range(Pi):
                    p = p_idx + 1
                    own_psi = (p - 1) * Ti + D[:, i]                     # (s,)
                    lower = own_psi
                    upper = own_psi + Ti

                    # --- Own deadline ---
                    w_own = _compute_w(
                        np.full((s, 1), p * Ci, dtype=np.float32),
                        own_psi.reshape(s, 1), p, Ci,
                        J, D, C, T, same_proc_i,
                    )
                    r_own = w_own[:, 0] - own_psi + D[:, i] + J[:, i]
                    max_r = r_own.copy()

                    # --- Interfering deadlines ---
                    for j in interfering:
                        Qj = int(Q_max_per_j[j])
                        Tj = T[j]
                        q_vals = np.arange(Qj, dtype=np.float32)

                        psi_j = (q_vals * Tj)[None, :] - J[:, j:j+1] + D[:, j:j+1]  # (s, Qj)
                        in_int = (psi_j >= lower[:, None]) & (psi_j < upper[:, None])
                        valid_q = (q_vals * Tj)[None, :] >= J[:, j:j+1] - 1e-4
                        mask = in_int & valid_q

                        if not mask.any():
                            continue

                        w_j = _compute_w(
                            np.full((s, Qj), p * Ci, dtype=np.float32),
                            psi_j, p, Ci,
                            J, D, C, T, same_proc_i,
                        )
                        r_j = w_j - psi_j + D[:, i:i+1] + J[:, i:i+1]
                        r_j_masked = np.where(mask, r_j, -1e9)
                        max_r = np.maximum(max_r, np.max(r_j_masked, axis=1))

                    # --- Pure deadlines (scalar adds {t.deadline}) ---
                    for j in interfering:
                        psi_dl = D[:, j]
                        in_int_dl = (psi_dl >= lower) & (psi_dl < upper)
                        if not in_int_dl.any():
                            continue

                        w_dl = _compute_w(
                            np.full((s, 1), p * Ci, dtype=np.float32),
                            psi_dl.reshape(s, 1), p, Ci,
                            J, D, C, T, same_proc_i,
                        )
                        r_dl = w_dl[:, 0] - psi_dl + D[:, i] + J[:, i]
                        max_r = np.where(in_int_dl, np.maximum(max_r, r_dl), max_r)

                    wcrt_i = np.maximum(wcrt_i, max_r)

                WCRT[:, i] = wcrt_i

                # Propagate jitter to immediate successors
                for j in range(n):
                    if pred_idx[j] == i:
                        J[:, j] = wcrt_i

            # Convergence / over-limit check
            changed = np.any(np.abs(WCRT - WCRT_prev) > 1e-4, axis=1)
            over_limit = np.any(WCRT > D_limit, axis=1)
            done = ~changed | over_limit

            for k in range(s):
                if done[k] and not cache.has_results(D_work[k]):
                    cache.insert(D_work[k], WCRT[k])

            if np.all(done):
                break

        # Cache remaining
        for k in range(s):
            if not cache.has_results(D_work[k]):
                cache.insert(D_work[k], WCRT[k])

        return _rebuild_from_cache(D_scenarios, cache)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, system: LinearSystem, scenarios_deadlines: np.ndarray = None):
        init_wcrt(system)
        C, T, D_task, D_flow, J_init, pred_idx, same_proc = _get_vectors(system)

        n = len(C)
        D_base = D_task.reshape(1, n).astype(np.float32)

        if scenarios_deadlines is not None and scenarios_deadlines.shape[0] > 0:
            D_all = np.concatenate([D_base, scenarios_deadlines], axis=0)
        else:
            D_all = D_base

        R = self._analysis(D_all, D_flow, C, T, J_init, pred_idx, same_proc)

        for task, wcrt in zip(system.tasks, R[:, 0]):
            task.wcrt = float(wcrt)

        self._full_response_times = R
        s_extra = 0 if scenarios_deadlines is None else scenarios_deadlines.shape[0]
        self._scenarios_response_times = R[:, 1:] if s_extra > 0 else None


def _topological_order(n, pred_idx):
    """Return tasks in topological order (predecessors before successors)."""
    processed = set()
    order = []

    def visit(idx):
        if idx in processed:
            return
        pi = pred_idx[idx]
        if pi >= 0 and pi not in processed:
            visit(pi)
        processed.add(idx)
        order.append(idx)

    for i in range(n):
        visit(i)
    return order


def _rebuild_from_cache(D_scenarios, cache):
    s = D_scenarios.shape[0]
    n = D_scenarios.shape[1]
    result = np.zeros((n, s), dtype=np.float32)
    for k in range(s):
        r = cache.get(D_scenarios[k])
        if r is not None:
            result[:, k] = r
    return result


# =========================================================================
# Deadline scenario builder
# =========================================================================

class DeadlineScenarios:
    def apply(self, system: LinearSystem, inputs: [[float]]) -> np.ndarray:
        max_d = max(t.deadline for t in system.tasks)
        n = len(system.tasks)
        s = len(inputs)
        D = np.zeros((s, n), dtype=np.float32)
        for k, x in enumerate(inputs):
            for i, v in enumerate(x[:n]):
                D[k, i] = v * max_d
        return D


# =========================================================================
# Gradient function
# =========================================================================

class VectorEDFGradientFunctionV2(GradientFunction):
    def __init__(self, sigma=1.5, cost_limit_factor=10):
        self.delta_function = AvgSeparationDelta(sigma=sigma)
        self.cost_limit_factor = cost_limit_factor
        self.scenarios_builder = DeadlineScenarios()
        self.cache = ResultsCache()

    def reset(self):
        self.cache.reset()

    def _compute_costs(self, system, inputs):
        tasks = system.tasks
        n = len(tasks)
        D_scenarios = self.scenarios_builder.apply(system, inputs)
        analysis = VectorHolisticEDFAnalysisV2(
            limit_factor=self.cost_limit_factor, cache=self.cache
        )
        analysis.apply(system, scenarios_deadlines=D_scenarios)
        R = analysis.scenarios_response_times
        if R is None:
            return np.zeros(len(inputs), dtype=np.float32)

        # Cost = max over FLOWS of (last_task_wcrt - flow_deadline) / flow_deadline
        # (matching InvslackCost behaviour)
        last_idx = [i for i, t in enumerate(tasks) if t.is_last]
        if not last_idx:
            return np.zeros(len(inputs), dtype=np.float32)

        R_last = R[last_idx, :]                                          # (n_flows, s)
        D_flow = np.array([tasks[i].flow.deadline for i in last_idx],
                          dtype=np.float32).reshape(-1, 1)               # (n_flows, 1)
        slack = (R_last - D_flow) / np.maximum(D_flow, 1e-9)             # (n_flows, s)
        costs = np.max(slack, axis=0)                                    # (s,)
        return costs

    def compute(self, system, x):
        deltas = self.delta_function.apply(system, x)
        inputs = gradient_inputs_from_deltas(x, deltas)
        costs = self._compute_costs(system, inputs)
        gradient = gradient_from_costs(costs, deltas)
        return gradient
