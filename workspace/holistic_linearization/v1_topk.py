"""V1-TopK: non-iterative priority assignment via vectorised V1 exploration.

Algorithm:
    1. Generate N candidate priority assignments per processor:
       - 1 DM assignment as anchor (never regress)
       - N-1 random permutations sampled uniformly per processor
    2. Score all N candidates in a single batched ``VectorLinearizedV1Analysis``
       (linearised surrogate, Spearman 0.97 vs Holistic).
    3. Keep the K candidates with lowest V1 cost (avg flow WCRT).
    4. Validate the K survivors with the full Holistic FP analysis.
       Return the first schedulable (or the best one if none is schedulable).

Total work per system: N V1-evals + K Holistic-evals (typically 100 + 5),
replacing GDPA's ~30 steps x 24 scenarios = 720 Holistic-evals.
"""
import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from model.analysis_function import init_wcrt, reset_wcrt
from model.linear_system import LinearSystem
from vector.vector_fp import PrioritiesMatrix

from workspace.holistic_linearization.vector_v1 import (
    VectorLinearizedV1Analysis,
)


def validate(system: LinearSystem) -> bool:
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return all(f.is_schedulable() for f in system.flows)


def v1_topk_assign(
    system: LinearSystem,
    n_candidates: int = 100,
    k: int = 5,
    limit_factor: int = 10,
    seed: int | None = None,
    perturbations: int = 0,
):
    """Run V1-TopK on ``system``.  Mutates system priorities in-place.

    Parameters
    ----------
    n_candidates : int
        Number of candidate priority assignments sampled.
    k : int
        Number of survivors validated with full Holistic.
    perturbations : int
        If 0, sample uniform random priorities per processor.  If >0,
        anchor candidates on DM and apply that many random adjacent swaps
        within each processor; this biases the search towards the DM
        baseline and dramatically increases the chance of finding good
        permutations for hard systems.
    """
    # DM baseline (anchor) — guarantees no regression vs DM
    PDAssignment(normalize=True).apply(system)
    n = len(system.tasks)
    base = [float(t.priority) for t in system.tasks]
    rng = np.random.default_rng(
        seed if seed is not None
        else (hash(system.name) % (2**32) if system.name else 0)
    )

    # ---- build N candidate priority vectors --------------------------------
    task_index = {t: i for i, t in enumerate(system.tasks)}
    candidates = [base]  # candidate 0 is the DM anchor
    if perturbations > 0:
        for _ in range(n_candidates - 1):
            cand = list(base)
            for proc in system.processors:
                idxs = [task_index[t] for t in proc.tasks]
                prio = [cand[idx] for idx in idxs]
                for _ in range(perturbations):
                    a = rng.integers(0, max(1, len(idxs) - 1))
                    b = min(a + 1, len(idxs) - 1)
                    prio[a], prio[b] = prio[b], prio[a]
                for j, idx in enumerate(idxs):
                    cand[idx] = float(prio[j])
            candidates.append(cand)
    else:
        for _ in range(n_candidates - 1):
            cand = list(base)
            for proc in system.processors:
                idxs = [task_index[t] for t in proc.tasks]
                vals = rng.random(len(idxs)) * 10.0
                for j, idx in enumerate(idxs):
                    cand[idx] = float(vals[j])
            candidates.append(cand)

    # ---- vectorised V1 ranking ---------------------------------------------
    pm = PrioritiesMatrix().apply(system, candidates)         # (N, t, t)
    reset_wcrt(system); init_wcrt(system)
    v1 = VectorLinearizedV1Analysis(limit_factor=limit_factor,
                                    max_p=1000)
    v1.apply(system, scenarios=pm)
    r = v1.scenarios_response_times                          # (t, N)

    # avg flow WCRT per scenario = mean over flow last-task WCRTs
    flow_last_idx = [task_index[f.tasks[-1]] for f in system.flows]
    costs = r[flow_last_idx, :].mean(axis=0)                 # (N,)

    # ---- top-K survivors + Holistic validation -----------------------------
    k = min(k, n_candidates)
    top_k_idx = np.argsort(costs)[:k]

    best_idx = int(top_k_idx[0])
    for idx in top_k_idx:
        cand = candidates[int(idx)]
        for t, p in zip(system.tasks, cand):
            t.priority = float(p)
        if validate(system):
            return True

    # leave system in best-V1 assignment, validate truthfully
    best_cand = candidates[best_idx]
    for t, p in zip(system.tasks, best_cand):
        t.priority = float(p)
    return validate(system)


# ---------------------------------------------------------------------------
# Adapter for SchedRatioEval — signature ``f(system) -> bool``
# ---------------------------------------------------------------------------

def make_v1_topk_method(n_candidates=100, k=5, limit_factor=10, perturbations=0):
    def _method(system: LinearSystem) -> bool:
        return v1_topk_assign(
            system,
            n_candidates=n_candidates,
            k=k,
            limit_factor=limit_factor,
            perturbations=perturbations,
        )
    return _method