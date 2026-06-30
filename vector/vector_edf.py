"""Vectorized Holistic Local EDF analysis.

Batch-processes multiple deadline scenarios simultaneously using a
discretised psi grid, enabling fast evaluation inside gradient-descent
(finite-difference) optimisation.

Architecture mirrors vector_fp.py:  scenarios × tasks × (p, psi_grid).
Exact ceil / floor arithmetic is used — the grid discretisation is the
only source of approximation (error < 1 job of interference).
"""

import numpy as np

from model.analysis_function import init_wcrt
from gradient_descent.gradient_function import AvgSeparationDelta, gradient_inputs_from_deltas, gradient_from_costs
from gradient_descent.interfaces import GradientFunction
from model.linear_system import LinearSystem


# =========================================================================
# Results cache
# =========================================================================

class ResultsCache:
    """Caches WCRT results keyed by deadline vectors."""

    def __init__(self):
        self.data = dict()

    @staticmethod
    def _key(deadlines: np.ndarray):
        return deadlines.tobytes()

    def insert(self, deadlines: np.ndarray, results: np.ndarray):
        key = self._key(deadlines)
        if key not in self.data:
            self.data[key] = results

    def get(self, deadlines: np.ndarray):
        key = self._key(deadlines)
        return self.data[key] if key in self.data else None

    def has_results(self, deadlines: np.ndarray):
        return self._key(deadlines) in self.data

    def reset(self):
        self.data.clear()

    def __len__(self):
        return len(self.data)


# =========================================================================
# Tensor builders
# =========================================================================

def _get_vectors(system: LinearSystem):
    """Extract task parameters as numpy vectors.  Returns (n,) or (n, 1) arrays."""
    tasks = system.tasks
    n = len(tasks)
    dtype = np.float32

    C = np.array([t.wcet for t in tasks], dtype=dtype).reshape(n, 1)
    T = np.array([t.period for t in tasks], dtype=dtype).reshape(n, 1)
    D_task = np.array([t.deadline for t in tasks], dtype=dtype).reshape(n, 1)
    D_flow = np.array([t.flow.deadline for t in tasks], dtype=dtype).reshape(n, 1)
    J_init = np.zeros((n, 1), dtype=dtype)

    t2i = {t: i for i, t in enumerate(tasks)}
    pred_idx = np.full(n, -1, dtype=np.int32)
    for i, t in enumerate(tasks):
        if t.predecessors:
            pi = t2i[t.predecessors[0]]
            pred_idx[i] = pi
            J_init[i, 0] = C[pi, 0]

    same_proc = np.zeros((n, n), dtype=np.bool_)
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            same_proc[i, j] = (ti.processor == tj.processor)

    return C, T, D_task, D_flow, J_init, pred_idx, same_proc


def _busy_period(C, T, J, same_proc, max_iters=200, tol=1e-6):
    """Compute busy periods L_i (n, 1) — exact, deadline-independent."""
    n = C.shape[0]
    L = C.copy()
    j_mask = same_proc & ~np.eye(n, dtype=bool)
    for _ in range(max_iters):
        own = np.ceil(L / T) * C
        L_exp = L.reshape(1, n)
        J_exp = J.reshape(1, n)
        T_exp = T.reshape(1, n)
        C_exp = C.reshape(1, n)
        interference = np.sum(
            np.ceil((L_exp + J_exp) / T_exp) * C_exp * j_mask,
            axis=1, keepdims=True
        )
        L_new = own + interference
        if np.allclose(L_new, L, atol=tol):
            return L_new
        L = L_new
    return L


# =========================================================================
# Vectorized Holistic EDF Analysis
# =========================================================================

class VectorHolisticEDFAnalysis:
    """Vectorized Holistic Local EDF schedulability analysis.

    Evaluates one base system + optionally many deadline-perturbed
    scenarios in a single batched tensor operation using a fixed psi grid.

    Parameters
    ----------
    M_psi : int
        Number of psi grid points per job *p*.
    limit_factor : float
        WCRT limit = limit_factor * deadline.
    cache : ResultsCache
        Optional cache to skip already-known deadline vectors.
    """

    def __init__(self, M_psi=100, limit_factor=10, cache=None):
        self.M_psi = M_psi
        self.limit_factor = limit_factor
        self.cache = cache if cache is not None else ResultsCache()
        self._scenarios_response_times = None
        self._full_response_times = None

    def clear_results(self):
        self._scenarios_response_times = None
        self._full_response_times = None

    @property
    def scenarios_response_times(self):
        """2D matrix (t, s) of response times for additional scenarios."""
        return self._scenarios_response_times

    @property
    def full_response_times(self):
        """2D matrix (t, s+1) including the base system."""
        return self._full_response_times

    # ------------------------------------------------------------------
    # Core batched analysis
    # ------------------------------------------------------------------

    def _analysis(self, D_scenarios, D_flow_sys, C, T, J_init, pred_idx, same_proc):
        """Run vectorised EDF analysis on *s* deadline scenarios.

        Parameters
        ----------
        D_scenarios : (s, n) float32
            Per-task deadlines for each scenario (EDF scheduling).
        D_flow_sys : (n,) float32
            Flow deadlines for the base system (limit check).
        C, T : (n, 1) float32
        J_init : (n, 1) float32
        pred_idx : (n,) int32
        same_proc : (n, n) bool

        Returns
        -------
        R : (n, s) float32
        """
        s = D_scenarios.shape[0]          # number of scenarios
        n = C.shape[0]                    # number of tasks
        M = self.M_psi
        limit = self.limit_factor
        cache = self.cache

        # Remove already-cached scenarios
        keep = np.ones(s, dtype=bool)
        for k in range(s):
            if cache.has_results(D_scenarios[k]):
                keep[k] = False
        D_work = D_scenarios[keep]        # (s', n)
        s_work = D_work.shape[0]

        if s_work == 0:
            # All cached — rebuild from cache
            return _rebuild_from_cache(D_scenarios, cache)

        # --- Busy periods and p-values ---
        L = _busy_period(C, T, J_init, same_proc)          # (n, 1)
        P_vals = np.ceil(L / T).astype(np.int32).ravel()   # (n,)
        P_vals = np.maximum(P_vals, 1)
        P_max = int(P_vals.max())
        p_idx = np.arange(1, P_max + 1, dtype=np.float32)  # (P_max,)

        # Validity mask: (n, P_max)
        p_valid = np.arange(P_max)[np.newaxis, :] < P_vals[:, np.newaxis]

        # --- Build psi grid: (s', n, P_max, M) ---
        # psi[s, i, p, m] = (p-1)*T_i + D_i[s] + (m/M)*T_i
        p_base = (p_idx[np.newaxis, np.newaxis, :, np.newaxis] - 1.0)  # (1, 1, P_max, 1)
        T_4d = T.reshape(1, n, 1, 1)                                     # (1, n, 1, 1)
        D_4d = D_work.reshape(s_work, n, 1, 1)                           # (s', n, 1, 1)
        m_frac = (np.arange(M, dtype=np.float32) / M).reshape(1, 1, 1, M)

        psi = p_base * T_4d + D_4d + m_frac * T_4d                       # (s', n, P_max, M)

        # --- Initialise working tensors ---
        # w[s, i, p, m] = p * C_i
        C_4d = C.reshape(1, n, 1, 1)                                     # (1, n, 1, 1)
        p_4d = p_idx.reshape(1, 1, P_max, 1)                             # (1, 1, P_max, 1)
        w = np.zeros((s_work, n, P_max, M), dtype=np.float32)
        w[:] = p_4d * C_4d                                                # broadcast to full shape

        r_max = np.zeros((s_work, n), dtype=np.float32)
        r_max_prev = r_max.copy()

        # Jitter: (s', n)
        J = np.tile(J_init.ravel(), (s_work, 1)).astype(np.float32)

        # Deadlines for EDF scheduling: (s', n)
        D_flat = D_work.astype(np.float32)

        # --- Predecessor indices for jitter updates ---
        pred_valid = pred_idx >= 0

        # --- Outer convergence loop ---
        while True:
            r_max_prev = r_max.copy()

            # ---------- w convergence ----------
            w_prev = np.empty_like(w)
            max_w_iters = 200
            for _ in range(max_w_iters):
                w_prev[:] = w

                # Broadcast for j-dimension: (s', n, n, P_max, M)
                w_brd = w[:, :, np.newaxis, :, :]                        # (s', 1, n, P_max, M)
                J_brd = J[:, np.newaxis, :, np.newaxis, np.newaxis]      # (s', n, 1, 1, 1) → (s', 1, n, 1, 1)
                T_brd = T.ravel().reshape(1, 1, n, 1, 1)                 # (1, 1, n, 1, 1)
                C_brd = C.ravel().reshape(1, 1, n, 1, 1)
                D_brd = D_flat[:, np.newaxis, :, np.newaxis, np.newaxis] # (s', 1, n, 1, 1)
                psi_brd = psi[:, :, np.newaxis, :, :]                     # (s', n, 1, P_max, M)

                # pl = max(0, ceil((w + J_j) / T_j))
                pl = np.maximum(0.0, np.ceil((w_brd + J_brd) / T_brd))

                # pd = (psi >= D_j) * max(0, floor((J_j + psi - D_j)/T_j) + 1)
                cond = (psi_brd >= D_brd).astype(np.float32)
                raw_pd = np.maximum(0.0, np.floor((J_brd + psi_brd - D_brd) / T_brd) + 1.0)
                pd = cond * raw_pd

                # Wi = min(pl, pd) * C_j   →  (s', n, n, P_max, M)
                Wi = np.minimum(pl, pd) * C_brd

                # Mask: same processor, exclude self
                sp_mask = same_proc.astype(np.float32)                  # (n, n)
                np.fill_diagonal(sp_mask, 0.0)
                sp_mask = sp_mask.reshape(1, n, n, 1, 1)                # (1, n, n, 1, 1)

                interference = np.sum(Wi * sp_mask, axis=2)              # (s', n, P_max, M)

                w_new = p_4d * C_4d + interference
                w = w_new

                if np.allclose(w, w_prev, atol=1e-4):
                    break

            # ---------- Response times ----------
            J_4d = J[:, :, np.newaxis, np.newaxis]                       # (s', n, 1, 1)
            r = w - psi + D_4d + J_4d                                    # (s', n, P_max, M)

            # Mask invalid (i, p)
            p_mask_4d = p_valid[np.newaxis, :, :, np.newaxis].astype(np.float32)  # (1, n, P_max, 1)
            r_masked = r * p_mask_4d - 1e9 * (1.0 - p_mask_4d)

            # Max over p and m
            r_flat = r_masked.reshape(s_work, n, -1)                    # (s', n, P_max*M)
            r_max_new = np.max(r_flat, axis=2)                           # (s', n)

            # Update r_max
            r_max = np.maximum(r_max, r_max_new)

            # --- Check over-limit scenarios ---
            D_limit = D_flow_sys.reshape(1, n) * limit                   # (1, n) — same for all scenarios
            over = np.any(r_max > D_limit, axis=1)                       # (s',)
            if np.any(over):
                # Cache over-limit results
                for k in np.where(over)[0]:
                    cache.insert(D_work[k], r_max[k])
                # Remove from working set
                keep_local = ~over
                D_work = D_work[keep_local]
                s_work = D_work.shape[0]
                if s_work == 0:
                    break
                # Slice all working tensors
                D_flat = D_flat[keep_local]
                J = J[keep_local]
                r_max = r_max[keep_local]
                r_max_prev = r_max_prev[keep_local]
                psi = psi[keep_local]
                w = w[keep_local]
                D_4d = D_work.reshape(s_work, n, 1, 1)
                J_4d = J[:, :, np.newaxis, np.newaxis]

            # --- Check converged scenarios ---
            converged = np.all(np.abs(r_max - r_max_prev) < 1e-4, axis=1)  # (s',)
            if np.any(converged):
                for k in np.where(converged)[0]:
                    cache.insert(D_work[k], r_max[k])
                keep_local = ~converged
                D_work = D_work[keep_local]
                s_work = D_work.shape[0]
                if s_work == 0:
                    break
                D_flat = D_flat[keep_local]
                J = J[keep_local]
                r_max = r_max[keep_local]
                r_max_prev = r_max_prev[keep_local]
                psi = psi[keep_local]
                w = w[keep_local]
                D_4d = D_work.reshape(s_work, n, 1, 1)
                J_4d = J[:, :, np.newaxis, np.newaxis]
                continue

            # --- Update jitter from predecessors ---
            J_new = np.zeros_like(J)
            J_new[:, :] = J_init.ravel()[np.newaxis, :]                  # reset to initial
            if np.any(pred_valid):
                J_new[:, pred_valid] = r_max[:, pred_idx[pred_valid]]
            J = J_new
            J_4d = J[:, :, np.newaxis, np.newaxis]

        # --- Rebuild full results from cache ---
        return _rebuild_from_cache(D_scenarios, cache)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(self, system: LinearSystem, scenarios_deadlines: np.ndarray = None):
        """Run vectorised EDF analysis.

        Parameters
        ----------
        system : LinearSystem
            The system to analyse (deadlines from the system are scenario 0).
        scenarios_deadlines : (s, n) float32 or None
            Additional deadline scenarios.
        """
        init_wcrt(system)
        C, T, D_task_sys, D_flow_sys, J_init, pred_idx, same_proc = _get_vectors(system)

        n = C.shape[0]
        D_base = D_task_sys.ravel().astype(np.float32).reshape(1, n)

        if scenarios_deadlines is not None and scenarios_deadlines.shape[0] > 0:
            D_all = np.concatenate([D_base, scenarios_deadlines], axis=0)
        else:
            D_all = D_base

        # Run analysis (D_flow_sys.ravel() for limit check)
        R = self._analysis(D_all, D_flow_sys.ravel(), C, T, J_init, pred_idx, same_proc)

        # Set system WCRTs from first column (base scenario)
        for task, wcrt in zip(system.tasks, R[:, 0]):
            task.wcrt = float(wcrt)

        self._full_response_times = R
        s_extra = 0 if scenarios_deadlines is None else scenarios_deadlines.shape[0]
        self._scenarios_response_times = R[:, 1:] if s_extra > 0 else None


def _rebuild_from_cache(D_scenarios, cache):
    """Reconstruct (n, s) result matrix from cache."""
    s = D_scenarios.shape[0]
    n = D_scenarios.shape[1]
    result = np.zeros((n, s), dtype=np.float32)
    for k in range(s):
        r = cache.get(D_scenarios[k])
        if r is not None:
            result[:, k] = r
    return result


# =========================================================================
# DeadlineScenario builder
# =========================================================================

class DeadlineScenarios:
    """Builds deadline matrices for gradient-descent finite-difference inputs.

    Each scenario is a slightly perturbed copy of the base-system deadlines.
    """

    def apply(self, system: LinearSystem, inputs: [[float]]) -> np.ndarray:
        """Convert a list of parameter vectors to deadline matrices.

        Parameters
        ----------
        system : LinearSystem
        inputs : list of list[float]
            Each inner list is a DeadlineExtractor-compatible parameter vector.

        Returns
        -------
        (s, n) float32 array of deadlines.
        """
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

class VectorEDFGradientFunction(GradientFunction):
    """Gradient function using vectorised EDF analysis for finite differences.

    Builds all perturbed deadline scenarios at once and evaluates them
    in a single batched vector operation.
    """

    def __init__(self, sigma=1.5, M_psi=100, cost_limit_factor=10):
        self.delta_function = AvgSeparationDelta(sigma=sigma)
        self.M_psi = M_psi
        self.cost_limit_factor = cost_limit_factor
        self.scenarios_builder = DeadlineScenarios()
        self.cache = ResultsCache()

    def reset(self):
        self.cache.reset()

    def _compute_costs(self, system: LinearSystem, inputs: [[float]]) -> np.ndarray:
        tasks = system.tasks
        n = len(tasks)
        D_scenarios = self.scenarios_builder.apply(system, inputs)      # (s, n)
        analysis = VectorHolisticEDFAnalysis(
            M_psi=self.M_psi, limit_factor=self.cost_limit_factor, cache=self.cache
        )
        analysis.apply(system, scenarios_deadlines=D_scenarios)
        R = analysis.scenarios_response_times                          # (n, s)
        if R is None:
            return np.zeros(len(inputs), dtype=np.float32)
        D_ref = np.max(D_scenarios, axis=1)                            # (s,) — conservative
        # Cost per scenario: max_i ( (R_i - D_i) / D_i )
        D_mat = D_scenarios.T                                          # (n, s)
        slack = (R - D_mat) / np.maximum(D_mat, 1e-9)
        costs = np.max(slack, axis=0)                                  # (s,)
        return costs

    def compute(self, system: LinearSystem, x: [float]) -> [float]:
        deltas = self.delta_function.apply(system, x)
        inputs = gradient_inputs_from_deltas(x, deltas)
        costs = self._compute_costs(system, inputs)
        gradient = gradient_from_costs(costs, deltas)
        return gradient
