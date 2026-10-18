"""Local-EDF deadline assignment by vectorized multi-start local search.

Scores a **whole neighbourhood** of candidate deadline vectors in a single call
to the (bounded) vectorized holistic local-EDF analysis:

* the neighbourhood is every single-coordinate multiplicative perturbation of the
  current assignment over a fine geometric grid;
* the best strictly improving move is taken and the sweep repeats until
  convergence (best-improvement coordinate descent);
* to escape local optima it runs an iterated local search: restarts from a
  perturbed incumbent or a fresh noisy analytic assignment, keeping the global
  best, all under a wall-clock budget;
* the objective is the judge's own quantity, the largest flow response ratio
  ``max_f R_f / D_f`` (plus a tiny sum tie-break), stopping at ``<= 1``.

The analytic starts are the family ``D_i = C_i * (D_flow / P_flow)^a / U_flow^d``.
The scoring analysis is :class:`vector.vector_edf.VectorHolisticEDFAnalysis`
(bit-identical to the scalar analysis on converging scenarios), run with an
iteration cap so it cannot be made pathologically slow by an infeasible
candidate.

Ported from the sandbox agent's ``vec_ils`` (generalized to any flow layout).
"""

import time

import numpy as np

from model.analysis_function import AnalysisFunction
from model.linear_system import LinearSystem
from utils.exec_time import ExecTime
from vector.vector_edf import VectorHolisticEDFAnalysis

# Wall-clock budget for the search, and a hard cap on the whole call (seconds).
IMPROVE_BUDGET = 30.0
MAX_TOTAL = 48.0
# Fine geometric grid of multiplicative coordinate moves.
FACTORS = np.exp(np.linspace(np.log(0.30), np.log(3.0), 18))
# Sweep cap of the bounded scoring analysis.
ITCAP = 1000


def _last_indices(system: LinearSystem) -> np.ndarray:
    """Indices (in ``system.tasks`` order) of the last task of each flow."""
    idx = []
    i = 0
    for flow in system.flows:
        i += len(flow.tasks)
        idx.append(i - 1)
    return np.array(idx)


class VecILSAssignment(AnalysisFunction):
    def __init__(self, improve_budget=IMPROVE_BUDGET, max_total=MAX_TOTAL,
                 factors=FACTORS, itcap=ITCAP):
        self.improve_budget = improve_budget
        self.max_total = max_total
        self.factors = np.asarray(factors, dtype=np.float64)
        self.analysis = VectorHolisticEDFAnalysis(itcap=itcap)
        self.exec_time = ExecTime()

    @staticmethod
    def _deadlines(system: LinearSystem) -> np.ndarray:
        return np.array([t.deadline for t in system.tasks], dtype=np.float64)

    @staticmethod
    def _set_deadlines(system: LinearSystem, D: np.ndarray) -> None:
        for t, x in zip(system.tasks, D):
            t.deadline = float(x)

    @staticmethod
    def _flow_deadlines(system: LinearSystem) -> np.ndarray:
        return np.array([t.flow.deadline for t in system.tasks], dtype=np.float64)

    @staticmethod
    def _score(W: np.ndarray, F: np.ndarray, last: np.ndarray) -> np.ndarray:
        r = W[:, last] / F[last]
        return r.max(axis=1) + 1e-6 * r.sum(axis=1)

    @staticmethod
    def _analytic(system: LinearSystem, a: float, d: float) -> np.ndarray:
        D = np.empty(len(system.tasks))
        i = 0
        for flow in system.flows:
            ratio = flow.deadline / flow.period
            uf = sum(t.wcet for t in flow.tasks) / flow.period
            for t in flow.tasks:
                D[i] = t.wcet * (ratio ** a) / (uf ** d)
                i += 1
        return D

    def _candidates(self, D: np.ndarray) -> np.ndarray:
        n = len(D)
        nf = len(self.factors)
        base = np.repeat(D[None, :], nf, axis=0)
        out = np.empty((n, nf, n))
        for i in range(n):
            blk = base.copy()
            blk[:, i] *= self.factors
            out[i] = blk
        return out.reshape(n * nf, n)

    def _descend(self, system, F, last, D, t_end):
        """Best-improvement coordinate descent with batched neighbourhood scoring."""
        sc = self._score(self.analysis.apply(system, D[None, :]), F, last)[0]
        while time.perf_counter() < t_end:
            cand = np.vstack([D[None, :], self._candidates(D)])
            s = self._score(self.analysis.apply(system, cand), F, last)
            j = int(np.argmin(s))
            if s[j] < sc - 1e-12:
                sc = s[j]
                D = cand[j].copy()
                if sc <= 1.0:
                    return D, sc
            else:
                break
        return D, sc

    def _improve(self, system, F, last, t_end, seed):
        rng = np.random.default_rng(seed)

        # Candidate starts in one batch: an analytic grid, plus the current
        # assignment when it is a valid (positive) warm start.
        starts = [self._analytic(system, a, d)
                  for a in (1.2, 1.5, 1.75, 2.0, 2.3)
                  for d in (0.3, 0.6, 1.0, 1.4)]
        cur = self._deadlines(system)
        if np.all(cur > 0):
            starts.insert(0, cur)
        S = self._score(self.analysis.apply(system, np.stack(starts)), F, last)
        order = np.argsort(S)
        best = starts[order[0]].copy()
        best_sc = S[order[0]]
        if best_sc <= 1.0:
            return best

        # Descend the most promising few starts.
        for j in order[:3]:
            if time.perf_counter() >= t_end:
                break
            D, sc = self._descend(system, F, last, starts[j], t_end)
            if sc < best_sc:
                best_sc, best = sc, D
                if best_sc <= 1.0:
                    return best

        # Alternating restarts: broad random analytic starts (global exploration)
        # and perturbations of the incumbent (local intensification).
        while time.perf_counter() < t_end and best_sc > 1.0:
            if rng.random() < 0.5:
                D0 = best.copy()
                k = int(rng.integers(1, 5))
                idx = rng.choice(len(D0), size=k, replace=False)
                D0[idx] *= np.exp(rng.normal(0, rng.uniform(0.05, 0.5), size=k))
            else:
                D0 = self._analytic(system, rng.uniform(1.0, 2.5), rng.uniform(0.2, 1.4))
                D0 = D0 * np.exp(rng.normal(0, rng.uniform(0.1, 0.5), size=len(D0)))
            D0 = np.clip(D0, 1e-9, F)
            D, sc = self._descend(system, F, last, D0, t_end)
            if sc < best_sc:
                best_sc, best = sc, D
        return best

    def apply(self, system: LinearSystem) -> LinearSystem:
        self.exec_time.init()
        t0 = time.perf_counter()
        budget = max(0.0, min(self.improve_budget,
                              self.max_total - (time.perf_counter() - t0)))
        seed = int(abs(sum(t.wcet for t in system.tasks)) * 1e6) % (2 ** 31)
        F = self._flow_deadlines(system)
        last = _last_indices(system)
        D = self._improve(system, F, last, time.perf_counter() + budget, seed)
        self._set_deadlines(system, D)
        self.exec_time.stop()
        return system
