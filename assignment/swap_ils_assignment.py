"""Fixed-priority assignment by swap-based iterated local search (ILS).

Searches over per-processor priority orderings directly, using a *smooth* proxy
objective -- the maximum end-to-end response-time ratio ``wcrt / flow.deadline``
-- evaluated with a relaxed response-time limit (``limit_factor > 1``) so that
near-feasible orderings still give a finite value and a usable gradient. The
neighbourhood is the set of same-processor transpositions; starting from the
proportional-deadline assignment it runs first-improvement descent, then
perturbs the ordering and descends again (iterated local search), stopping as
soon as a schedulable ordering is found. The whole system is re-evaluated after
every swap, so the cross-processor jitter coupling is taken into account
exactly.

Ported from the sandbox agent's ``swap_ils``.
"""

import random

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from model.analysis_function import AnalysisFunction
from model.linear_system import LinearSystem
from utils.exec_time import ExecTime

# Response-time limit for the search proxy (> 1: near-infeasible orderings still
# yield a finite objective). 2.0 is a good speed/precision compromise.
LIMIT_FACTOR = 2.0
# Perturb-and-descend restarts attempted when the trivial start is not schedulable.
RESTARTS = 64
# Deterministic seed (reproducible results).
SEED = 0x5EED


class SwapILSAssignment(AnalysisFunction):
    def __init__(self, limit_factor=LIMIT_FACTOR, restarts=RESTARTS, seed=SEED):
        self.limit_factor = limit_factor
        self.restarts = restarts
        self.seed = seed
        self.exec_time = ExecTime()

    def _proxy(self, system: LinearSystem) -> float:
        """Max normalized end-to-end WCRT; ``1e6`` if the analysis aborts."""
        HolisticFPAnalysis(limit_factor=self.limit_factor, reset=True).apply(system)
        worst = 0.0
        for flow in system.flows:
            wcrt = flow.wcrt
            ratio = wcrt / flow.deadline if wcrt is not None else 1e6
            if ratio > worst:
                worst = ratio
        return worst

    def _swap_descent(self, system: LinearSystem, best: float):
        """First-improvement local search over same-processor transpositions.

        Returns ``(found, best)``; ``found`` is True if a schedulable ordering
        (proxy cost < 1) was reached.
        """
        for _ in range(20):
            improved = False
            for proc in system.processors:
                tasks = proc.tasks
                n = len(tasks)
                for i in range(n):
                    ti = tasks[i]
                    for j in range(i + 1, n):
                        tj = tasks[j]
                        ti.priority, tj.priority = tj.priority, ti.priority
                        cost = self._proxy(system)
                        if cost < best - 1e-12:
                            best = cost
                            improved = True
                            if cost < 1.0:
                                return True, best
                        else:
                            ti.priority, tj.priority = tj.priority, ti.priority
            if not improved:
                break
        return False, best

    def apply(self, system: LinearSystem) -> LinearSystem:
        self.exec_time.init()

        # Trivial but strong start: proportional local deadlines (pd).
        PDAssignment(normalize=True).apply(system)

        best = self._proxy(system)
        best_snapshot = [task.priority for task in system.tasks]
        if best >= 1.0:
            rng = random.Random(self.seed)
            procs = system.processors
            for _ in range(self.restarts):
                # Perturb a random number of processors (1..3), then descend.
                k = rng.choice((1, 2, 3))
                for proc in rng.sample(procs, min(k, len(procs))):
                    order = list(range(len(proc.tasks)))
                    rng.shuffle(order)
                    for rank, idx in enumerate(order):
                        proc.tasks[idx].priority = rank + 1
                found, cost = self._swap_descent(system, self._proxy(system))
                if found:
                    break
                if cost < best:
                    best = cost
                    best_snapshot = [task.priority for task in system.tasks]
                # continue from the best solution found so far
                for task, prio in zip(system.tasks, best_snapshot):
                    task.priority = prio

        self.exec_time.stop()
        return system
