import numpy as np

from model.linear_system import LinearSystem, Task


def _higher_priority(task: Task) -> list[Task]:
    return [t for t in task.processor.tasks
            if t.priority >= task.priority and t != task]


def _init_wcrt(system: LinearSystem):
    for flow in system.flows:
        tks = flow.tasks
        for i, task in enumerate(tks):
            task.wcrt = task.wcet
            if i > 0:
                task.wcrt += tks[i - 1].wcrt


# ---------------------------------------------------------------------------
# Shared: per-task analysis kernel (linearised w + p loop)
# ---------------------------------------------------------------------------

def _analyse_task_linear(task: Task, limit_factor: float, max_p: int) -> None:
    """Compute WCRT for a single task using the linearised formula."""
    hp = _higher_priority(task)
    limit = task.flow.deadline * limit_factor
    T_i = task.period
    C_i = task.wcet
    J_i = task.jitter

    U_hp = sum(t.wcet / t.period for t in hp)
    denominator = 1.0 - U_hp
    if denominator <= 0:
        task.wcrt = float('inf')
        for t in task.all_successors:
            t.wcrt = float('inf')
        return

    JU_hp = sum(t.jitter * t.wcet / t.period for t in hp)

    p = 1
    while p <= max_p:
        w = (p * C_i + JU_hp) / denominator
        r = w - (p - 1) * T_i + J_i

        if r > task.wcrt:
            task.wcrt = r

        if r > limit:
            for t in task.all_successors:
                t.wcrt = task.wcrt
            return

        if w <= p * T_i:
            break

        p += 1


# ---------------------------------------------------------------------------
# V1 -- Convergent iteration  (alpha=0, p loop, full convergence)
# ---------------------------------------------------------------------------

class LinearizedFPAnalysis:
    """V1: Full convergence loop.

    Iterates over all tasks until WCRTs stabilise.  Each inner step
    uses the closed-form linearised *w* (alpha = 0 by default).

    Highest correlation with the original Holistic FP (Spearman ~ 0.94).
    """

    def __init__(self, limit_factor=10, alpha=0.0, max_p=100, verbose=False):
        self.limit_factor = limit_factor
        self.alpha = alpha
        self.max_p = max_p
        self.verbose = verbose

    def apply(self, system: LinearSystem) -> None:
        _init_wcrt(system)

        tasks = system.tasks
        wcrts = [t.wcrt for t in tasks]
        wcrts_prev = [0.0] * len(tasks)

        while wcrts != wcrts_prev:
            wcrts_prev = wcrts[:]

            for task in tasks:
                hp = _higher_priority(task)
                limit = task.flow.deadline * self.limit_factor
                T_i = task.period
                C_i = task.wcet
                J_i = task.jitter

                C_hp = sum(t.wcet for t in hp)
                U_hp = sum(t.wcet / t.period for t in hp)
                JU_hp = sum(t.jitter * t.wcet / t.period for t in hp)

                denominator = 1.0 - U_hp
                if denominator <= 0:
                    task.wcrt = float('inf')
                    for t in task.all_successors:
                        t.wcrt = float('inf')
                    return

                p = 1
                while p <= self.max_p:
                    w = (p * C_i + self.alpha * C_hp + JU_hp) / denominator
                    r = w - (p - 1) * T_i + J_i

                    if r > task.wcrt:
                        task.wcrt = r

                    if r > limit:
                        for t in task.all_successors:
                            t.wcrt = task.wcrt
                        return

                    if w <= p * T_i:
                        break

                    p += 1

            wcrts = [t.wcrt for t in tasks]


# ---------------------------------------------------------------------------
# V2 -- N-pass  (no convergence loop, alpha=0, p loop)
# ---------------------------------------------------------------------------

class LinearizedFPAnalysisV2:
    """V2: N-pass linearised FP (no outer convergence).

    Processes tasks *n_passes* times through the task list, each time
    using the freshest WCRT values for jitter.  No convergence check.

    Parameters
    ----------
    n_passes : int
        Number of passes over the full task set (default 2).
        Increase for better accuracy at the cost of speed.
    """

    def __init__(self, limit_factor=10, max_p=100, n_passes=2, verbose=False):
        self.limit_factor = limit_factor
        self.max_p = max_p
        self.n_passes = n_passes
        self.verbose = verbose

    def apply(self, system: LinearSystem) -> None:
        _init_wcrt(system)
        tasks = system.tasks

        for _pass in range(self.n_passes):
            for task in tasks:
                _analyse_task_linear(task, self.limit_factor, self.max_p)


# ---------------------------------------------------------------------------
# V3 -- One-shot linear system  (alpha=0, p=1, matrix solve)
# ---------------------------------------------------------------------------

class LinearizedFPAnalysisV3:
    """V3: One-shot algebraic solve (p = 1).

    Sets up the fixed-point equations

        r_i = C_i/D_i + r_{pred(i)} + (1/D_i) * sum_{j in hp(i)} u_j * r_{pred(j)}

    and solves  (I - M) r = b  in a single numpy call.

    **Limitation**: assumes p = 1 for all tasks; underestimates WCRT
    when a task's response exceeds its period.  Best used for
    lightly-loaded systems or when the formula needs to be embedded
    as linear constraints.
    """

    def __init__(self, limit_factor=10, verbose=False):
        self.limit_factor = limit_factor
        self.verbose = verbose

    def apply(self, system: LinearSystem) -> None:
        tasks = system.tasks
        n = len(tasks)
        idx = {task: i for i, task in enumerate(tasks)}

        b = np.zeros(n)
        M = np.zeros((n, n))

        for i, task in enumerate(tasks):
            hp = _higher_priority(task)
            U_hp = sum(t.wcet / t.period for t in hp)
            D = 1.0 - U_hp

            if D <= 0:
                task.wcrt = float('inf')
                for t in task.all_successors:
                    t.wcrt = float('inf')
                b[i] = float('inf')
                continue

            b[i] = task.wcet / D

            preds = task.predecessors
            if preds:
                k = idx.get(preds[0])
                if k is not None:
                    M[i, k] = 1.0

            for h in hp:
                hp_preds = h.predecessors
                if hp_preds:
                    k = idx.get(hp_preds[0])
                    if k is not None:
                        M[i, k] += (h.wcet / h.period) / D

        I = np.eye(n)
        try:
            r_vec = np.linalg.solve(I - M, b)
        except np.linalg.LinAlgError:
            for task in tasks:
                task.wcrt = float('inf')
            return

        for i, task in enumerate(tasks):
            val = float(r_vec[i])
            if np.isfinite(val) and val > 0:
                task.wcrt = max(task.wcet, val)
            else:
                task.wcrt = float('inf')

            limit = task.flow.deadline * self.limit_factor
            if task.wcrt > limit:
                for t in task.all_successors:
                    t.wcrt = task.wcrt


# ---------------------------------------------------------------------------
# Legacy alias (backward compat)
# ---------------------------------------------------------------------------

LinearizedFPAnalysisV4 = LinearizedFPAnalysisV2
