"""Vectorised V1 linearised FP analysis.

Mirrors ``VectorHolisticFPAnalysis`` (vector/vector_fp.py) but replaces the
inner w-convergence loop by the linearised closed form

    w = (p * C_i + JU_hp) / (1 - U_hp)

so every p-iteration is a single numpy expression instead of a fixed-point
loop.  Together with batched evaluation of priority scenarios this makes the
V1 surrogate suitable for finite-difference gradient computation at a speed
comparable to GDPA.
"""
import numpy as np

from vector.vector_fp import (
    ResultsCache,
    system_priority_matrix,
    successor_matrix,
    get_vectors,
    converged_scenarios,
    scenarios_over_limit,
    cache_scenario_results,
    remove_scenarios,
    build_results_from_cache,
    prune_known_scenarios,
)
from model.analysis_function import init_wcrt
from model.linear_system import LinearSystem


def jitter_matrix_v1(sm: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Per-task jitter = max(predecessor wcrt); one predecessor/task assumed.

    ``sm`` is the (t, t) successor matrix; ``r`` is (s, t, 1).  Returns (s, t, 1)
    preserving dtype (no float32 downcast).
    """
    return (sm.T @ r).astype(r.dtype, copy=False)


class VectorLinearizedV1Analysis:
    """Vectorised V1 linearised FP analysis.

    API-compatible with ``VectorHolisticFPAnalysis``:
        ``apply(system, scenarios=None)``
        ``scenarios_response_times``  -> (t, s) matrix for extra scenarios
        ``full_response_times``       -> (t, s+1) matrix incl. input system
    """

    def __init__(self, verbose=False, limit_factor=10,
                 cache: ResultsCache = None, dtype=np.float64,
                 max_p: int = 1000):
        self.verbose = verbose
        self.limit_factor = limit_factor
        # avoid mutable-default-arg: a shared cache would leak results
        # across instances and cause premature pruning
        self.cache = cache if cache is not None else ResultsCache()
        self.dtype = dtype
        self.max_p = max_p
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

    # ------------------------------------------------------------------ #

    @staticmethod
    def _analysis(priority_matrix, wcets, periods, deadlines, successors,
                   wcrts, jitters, cache,
                   verbose=False, limit=10, dtype=np.float64, max_p=1000):
        assert wcets.shape == periods.shape == deadlines.shape \
            == successors.shape == wcrts.shape == jitters.shape
        assert wcets.shape[1] == 1
        assert priority_matrix.shape[1] == wcets.shape[0]

        # working priority matrix; drop scenarios whose results are cached
        pm = priority_matrix.copy()
        pm = prune_known_scenarios(pm, cache)
        s, t, _ = pm.shape
        if s == 0:
            return build_results_from_cache(priority_matrix, cache)

        # cast to float for matmul and keep the bool matrix in sync via pruning
        pm_f = pm.astype(dtype)

        # successor matrix (t, t) used for jitter propagation
        sm = successor_matrix(successors).astype(dtype)

        # static task data, broadcast appropriately
        C = wcets.astype(dtype).reshape(1, t, 1)
        T = periods.astype(dtype).reshape(1, t, 1)

        # per-task utilisation (t, 1)
        u = (wcets / periods).astype(dtype).reshape(t, 1)

        # U_hp[si,i] = sum_k pm[si,i,k] * u[k]        -> (s, t, 1)
        U_hp = pm_f @ u
        D = (1.0 - U_hp)
        valid_D = (D > 0)

        r_limit = limit * deadlines  # (t, 1)

        # init r_max from the system's chain-initialised wcrts
        r_max = wcrts.astype(dtype).reshape(1, t, 1).repeat(s, axis=0).copy()
        r = r_max.copy()
        r_max_prev = r_max.copy()

        # initial jitters from current r_max
        j = jitter_matrix_v1(sm, r_max)  # (s, t, 1)
        p_mask = np.full(r.shape, False)

        while pm.size > 0:
            r_max_prev = r_max.copy()
            p = 1
            p_mask = np.full(r.shape, False)
            while not np.all(p_mask):
                stop_p = p * T  # (1, t, 1)

                # J * u contribution per interferer.  ju_shape = (s, t, 1)
                ju = j * u.reshape(1, t, 1)            # (s, t, 1)

                # JU_hp[si,i] = sum_k pm[si,i,k] * ju[si,k,0]   -> (s, t, 1)
                # Matmul (s, t, t) @ (s, t, 1): interferer index k contracted.
                JU_hp = pm_f @ ju

                # closed-form w (mask disabled-D positions to inf)
                w = (p * C + JU_hp) / np.where(valid_D, D, 1.0)
                w = np.where(valid_D, w, np.inf)
                # don't update tasks that already converged their p-loop
                w = w * (~p_mask)
                # also disable their r contribution
                r_iter = w - (p - 1) * T + j
                r_iter = np.where(p_mask, r_max, r_iter)

                r_max = np.maximum(r_max, r_iter)

                # cache & prune scenarios that have crossed their WCRT limit
                over = scenarios_over_limit(r_max, r_limit)
                if np.any(over):
                    cache_scenario_results(r_max, pm, over, cache)
                    (pm, pm_f, r_max, r, r_max_prev, j, U_hp, D, valid_D,
                     p_mask, w) = remove_scenarios(
                        over, pm, pm_f, r_max, r, r_max_prev, j, U_hp, D,
                        valid_D, p_mask, w)
                    if pm.size == 0:
                        break

                # refresh jitters using the updated r_max
                j = jitter_matrix_v1(sm, r_max)

                p_mask = w <= stop_p
                if verbose:
                    print(f"V1vec p={p} pmask="
                          f"{int(np.sum(p_mask))}/{p_mask.size}")
                p += 1
                if p > max_p:
                    break

            converged = converged_scenarios(r_max, r_max_prev)
            if np.any(converged):
                cache_scenario_results(r_max, pm, converged, cache)
                (pm, pm_f, r_max, r, r_max_prev, j, U_hp, D, valid_D) = \
                    remove_scenarios(converged, pm, pm_f, r_max, r,
                                     r_max_prev, j, U_hp, D, valid_D)
                if verbose:
                    print(f"V1vec outer converged="
                          f"{int(np.sum(converged))}")

        return build_results_from_cache(priority_matrix, cache)

    # ------------------------------------------------------------------ #

    def apply(self, system: LinearSystem, scenarios: np.array = None):
        """Run the vectorised V1 analysis.

        ``scenarios`` is an optional 3D (s, t, t) batch of extra priority
        matrices.  Response times for the input system are written to
        ``task.wcrt``; the full (t, s+1) result matrix is also exposed via
        ``full_response_times``.
        """
        init_wcrt(system)
        wcets, periods, deadlines, successors, wcrts, jitters = get_vectors(
            system, single_precision=False)

        n = len(wcets)
        input_pm = system_priority_matrix(system).reshape(1, n, n)
        s = 0 if scenarios is None else scenarios.shape[0]
        pm = np.concatenate((input_pm, scenarios), axis=0) \
            if s > 0 else input_pm

        r = self._analysis(
            pm, wcets, periods, deadlines, successors, wcrts, jitters,
            self.cache,
            verbose=self.verbose, limit=self.limit_factor,
            dtype=self.dtype, max_p=self.max_p,
        )

        for task, wcrt in zip(system.tasks, r[:, 0]):
            task.wcrt = float(wcrt)

        self._full_response_times = r
        self._scenarios_response_times = r[:, 1:] if s > 0 else None