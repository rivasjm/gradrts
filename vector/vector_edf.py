"""Vectorized holistic local-EDF analysis (batch of deadline scenarios).

Vectorized counterpart of :class:`analysis.holistic_local_edf_analysis.HolisticLocalEDFAnalysis`.
It evaluates a whole ``(K, n_tasks)`` batch of local-deadline assignments **at
once** with numpy and reproduces the scalar analysis **bit for bit** (same
floating-point arithmetic: comparison-based window assignment and
CPython-compatible compensated summation).

``VectorHolisticEDFAnalysis.apply(system, scenarios_deadlines)`` returns the
per-task WCRTs of every assignment as an ``(K, n_tasks)`` array (``inf`` where
the analysis aborts). ``scenarios_deadlines[j][i]`` is the local deadline of task
``i`` (in ``system.tasks`` order) for assignment ``j``.

The class takes an optional ``itcap``: ``None`` (default) is the exact unbounded
analysis; an integer caps the outer Gauss-Seidel loop and drops scenarios that
abort or fail to converge within ``itcap`` sweeps (bit-exact with the unbounded
analysis on converging scenarios, and robust as a *search* objective).
"""

import numpy as np

_RTOL = 1e-9  # math.isclose default rel_tol


def _isclose(a, b):
    # math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0):
    #   |a - b| <= rel_tol * max(|a|, |b|)
    return np.abs(a - b) <= _RTOL * np.maximum(np.abs(a), np.abs(b))


# The reference's ``sum`` uses CPython's compensated (Neumaier) summation. The
# per-addend error is recovered inline with Knuth's branchless two-sum
# (``bb = t - hi; err = (hi - (t - bb)) + (x - bb)``), exact and equal to
# Neumaier's compensation, inlined in the hot loops for speed.


class _Kernel:
    """Batch-independent structure of one system (mapping + parameters)."""

    def __init__(self, system):
        tasks = system.tasks
        self.tasks = tasks
        self.n = n = len(tasks)
        pos = {id(t): i for i, t in enumerate(tasks)}

        self.P = np.array([t.period for t in tasks], dtype=np.float64)
        self.C = np.array([t.wcet for t in tasks], dtype=np.float64)
        self.F = np.array([t.flow.deadline for t in tasks], dtype=np.float64)

        self.pred = [None] * n
        for flow in system.flows:
            idx = [pos[id(t)] for t in flow.tasks]
            for a, b in zip(idx, idx[1:]):
                self.pred[b] = a

        # cotasks in the exact order of task.processor.tasks (minus self)
        self.cotask = [[pos[id(x)] for x in t.processor.tasks if x is not t]
                       for t in tasks]
        self.cot_P = [np.array([tasks[j].period for j in ker], dtype=np.float64)
                      for ker in self.cotask]
        self.cot_C = [np.array([tasks[j].wcet for j in ker], dtype=np.float64)
                      for ker in self.cotask]


def _task_max_r(ker, W, D, i, K):
    """Vectorized per-task kernel: returns max response ``(K,)`` for task ``i``.

    ``W`` is the current ``(K, n)`` WCRT cache; ``D`` is the ``(K, n)`` deadline
    batch.
    """
    P, C = ker.P, ker.C
    pred, cotask = ker.pred, ker.cotask
    Pi, Ci, Di = P[i], C[i], D[:, i]

    def jitter(j):
        p = pred[j]
        return np.zeros(K) if p is None else W[:, p]

    Ji = jitter(i)
    cot = cotask[i]
    m = len(cot)
    if m:
        Jt = np.empty((m, K))
        Pt = ker.cot_P[i]
        Ct = ker.cot_C[i]
        Dt = np.empty((m, K))
        for r, t in enumerate(cot):
            Jt[r] = jitter(t)
            Dt[r] = D[:, t]

    # --- busy period L (vector fixed point, reference expression) ----------
    L = np.full(K, Ci)
    bp_active = np.ones(K, dtype=bool)
    while bp_active.any():
        if m:
            hi = np.ceil((L + Jt[0]) / Pt[0]) * Ct[0]
            lo = 0.0
            for r in range(1, m):
                x = np.ceil((L + Jt[r]) / Pt[r]) * Ct[r]
                t = hi + x
                bb = t - hi
                lo = lo + ((hi - (t - bb)) + (x - bb))
                hi = t
            s = hi + lo
        else:
            s = 0.0
        nxt = np.ceil(L / Pi) * Ci + s
        done = bp_active & _isclose(nxt, L)
        L = np.where(bp_active, nxt, L)
        bp_active = bp_active & ~done

    # --- candidate psi points ----------------------------------------------
    psi_parts = []
    valid_parts = []

    q_own = np.ceil(L / Pi)
    Qo = max(1, int(q_own.max()))
    q = np.arange(1, Qo + 1, dtype=np.float64)
    psi_parts.append(Di[:, None] + (q[None, :] - 1.0) * Pi)
    valid_parts.append(q[None, :] <= q_own[:, None])

    for r in range(m):
        Pr, Cr, Dr, J = Pt[r], Ct[r], Dt[r], Jt[r]
        cnt = np.ceil((L + J) / Pr)
        Q = max(1, int(cnt.max()))
        q = np.arange(1, Q + 1, dtype=np.float64)
        off = (q[None, :] - 1.0) * Pr - J[:, None]
        psi_parts.append(off + Dr[:, None])
        valid_parts.append((q[None, :] <= cnt[:, None]) & (off >= 0.0))
        psi_parts.append(Dr[:, None])
        valid_parts.append(np.ones((K, 1), dtype=bool))

    psi = np.concatenate(psi_parts, axis=1)
    valid = np.concatenate(valid_parts, axis=1)

    # --- window index by comparison (bit-exact with the reference) ---------
    Pmax = np.ceil(L / Pi)
    Pmax_i = max(1, int(Pmax.max()))
    par = np.arange(1, Pmax_i + 2, dtype=np.float64)
    lower = (par[None, :] - 1.0) * Pi + Di[:, None]          # (K, Pmax_i+1)
    p = (lower[:, 0:Pmax_i, None] <= psi[:, None, :]).sum(axis=1)
    up_idx = np.minimum(Pmax.astype(np.int64), Pmax_i)
    upper = np.take_along_axis(lower, up_idx[:, None], axis=1)[:, 0]
    valid = valid & (p >= 1) & (psi < upper[:, None])
    p = np.where(valid, p.astype(np.float64), 1.0)
    psi_eff = np.where(valid, psi, Di[:, None])

    # --- w_ab fixed point ---------------------------------------------------
    pCi = p * Ci
    w = pCi
    w_active = np.ones((K, psi.shape[1]), dtype=bool)

    # pd's inputs do not depend on w; precompute them once.
    pd_num = [Jt[r][:, None] + psi_eff - Dt[r][:, None] for r in range(m)]
    pd_lt = [psi_eff < Dt[r][:, None] for r in range(m)]

    def _term(r):
        pl = np.ceil((w + Jt[r][:, None]) / Pt[r])
        pd = np.where(pd_lt[r], 0.0, np.floor(pd_num[r] / Pt[r]) + 1.0)
        return np.minimum(pl, pd) * Ct[r]

    while w_active.any():
        if m:
            hi = _term(0)
            lo = 0.0
            for r in range(1, m):
                x = _term(r)
                t = hi + x
                bb = t - hi
                lo = lo + ((hi - (t - bb)) + (x - bb))
                hi = t
            s = hi + lo
        else:
            s = 0.0
        acc = pCi + s
        done = w_active & _isclose(acc, w)
        w = np.where(w_active, acc, w)
        w_active = w_active & ~done

    r_resp = w - psi_eff + Di[:, None] + Ji[:, None]
    r_resp = np.where(valid, r_resp, -np.inf)
    return r_resp.max(axis=1)


def analyze(system, deadlines):
    """Exact vectorized analysis: ``(K, n_tasks)`` deadlines -> ``(K, n_tasks)`` WCRTs."""
    D = np.asarray(deadlines, dtype=np.float64)
    if D.ndim == 1:
        D = D[None, :]
    K = D.shape[0]

    ker = _Kernel(system)
    n = ker.n
    C = ker.C
    F = ker.F
    pred = ker.pred

    W = np.zeros((K, n))
    for i in range(n):
        W[:, i] = C[i] if pred[i] is None else C[i] + W[:, pred[i]]

    aborted = np.zeros(K, dtype=bool)

    # Tasks whose evaluation reads task j's WCRT: its processor-mates (local-EDF
    # interference) and its flow successor (release jitter).
    dependents = [None] * n
    for j in range(n):
        deps = set(ker.cotask[j])
        for f in range(n):
            if pred[f] == j:
                deps.add(f)
        dependents[j] = sorted(deps)

    # Gauss-Seidel with per-task dirty tracking (bit-identical to re-evaluating).
    dirty = np.ones((K, n), dtype=bool)

    while dirty.any():
        progressed = False
        for i in range(n):
            sel = np.nonzero(dirty[:, i])[0]
            if sel.size == 0:
                continue
            progressed = True
            sub_W = W[sel]
            sub_D = D[sel]
            max_r = _task_max_r(ker, sub_W, sub_D, i, sel.size)
            act = ~aborted[sel]
            over = (max_r > F[i] * 10.0) & act
            aborted[sel] |= over
            upd = (max_r > sub_W[:, i]) & act
            dirty[sel, i] = False
            if upd.any():
                sub_W[:, i] = np.where(upd, max_r, sub_W[:, i])
                W[sel] = sub_W
                rows = sel[upd]
                for d in dependents[i]:
                    dirty[rows, d] = True
        if not progressed:
            break

    return np.where(aborted[:, None], np.inf, W)


class VectorHolisticEDFAnalysis:
    """Batch interface (mirrors the scalar analysis API).

    ``apply(system, scenarios_deadlines)`` returns the per-task WCRTs of every
    assignment as an ``(K, n_tasks)`` array (``inf`` where it aborts). A single
    ``(n_tasks,)`` assignment is accepted and returned as ``(1, n_tasks)``.

    ``itcap`` selects the fixed-point sweep cap: ``None`` (default) is the exact
    unbounded analysis; an integer bounds the outer loop and drops scenarios that
    fail to converge within it (a robust *search* objective).
    """

    def __init__(self, itcap=None):
        self.itcap = itcap

    def apply(self, system, scenarios_deadlines):
        if self.itcap is None:
            return analyze(system, scenarios_deadlines)
        return analyze_bounded(system, scenarios_deadlines, self.itcap)

    __call__ = apply


def analyze_bounded(system, deadlines, itcap=1000):
    """Like :func:`analyze` but caps the outer fixed point at ``itcap`` sweeps.

    Scenarios still dirty at the cap (or aborted) are returned as ``inf``. On
    converging scenarios this is bit-identical to :func:`analyze`.
    """
    D = np.asarray(deadlines, dtype=np.float64)
    if D.ndim == 1:
        D = D[None, :]
    K = D.shape[0]

    ker = _Kernel(system)
    n = ker.n
    C = ker.C
    F = ker.F
    pred = ker.pred

    W = np.zeros((K, n))
    for i in range(n):
        W[:, i] = C[i] if pred[i] is None else C[i] + W[:, pred[i]]

    aborted = np.zeros(K, dtype=bool)

    dependents = [None] * n
    for j in range(n):
        deps = set(ker.cotask[j])
        for f in range(n):
            if pred[f] == j:
                deps.add(f)
        dependents[j] = sorted(deps)

    dirty = np.ones((K, n), dtype=bool)
    iters = 0
    while dirty.any() and iters < itcap:
        iters += 1
        progressed = False
        for i in range(n):
            sel = np.nonzero(dirty[:, i] & ~aborted)[0]
            if sel.size == 0:
                continue
            progressed = True
            sub_W = W[sel]
            max_r = _task_max_r(ker, sub_W, D[sel], i, sel.size)
            over = max_r > F[i] * 10.0
            if over.any():
                aborted[sel[over]] = True
                dirty[sel[over], i] = False
            keep = ~over
            if not keep.any():
                continue
            selk = sel[keep]
            mr = max_r[keep]
            wk = W[selk]
            dirty[selk, i] = False
            upd = mr > wk[:, i]
            if upd.any():
                wk[:, i] = np.where(upd, mr, wk[:, i])
                W[selk] = wk
                rows = selk[upd]
                for d in dependents[i]:
                    dirty[rows, d] = True
        if not progressed:
            break

    aborted |= dirty.any(axis=1)
    return np.where(aborted[:, None], np.inf, W)
