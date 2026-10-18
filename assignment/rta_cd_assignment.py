"""Local-EDF deadline assignment by response-time-guided coordinate descent.

The holistic local-EDF analysis turns a local-deadline assignment into per-flow
worst-case response times. This assignment:

1. starts from an analytic assignment that gives priority to flows whose
   end-to-end deadline is tight relative to their period
   ``D_i ~ C_i * (D_flow / P_flow)^a / U_flow^d``;
2. runs a coordinate descent over the local deadlines, using the analysis itself
   as the objective ``sum_f max(0, R_f/D_f - 1)``, with a relaxed
   ``limit_factor`` so hopeless states abort quickly;
3. short-circuits as soon as the true (``limit_factor=1``) analysis reports the
   system schedulable.

The descent stops deterministically after ``MAX_SWEEPS`` full sweeps, or earlier
when a sweep improves nothing. The wall-clock guard only abandons a sweep that is
taking pathologically long (it is a safety valve, not the stopping rule), and the
number of instances that reach it is reported by the runners.

The result is never worse than the analytic starting point.

Ported from the sandbox agent's ``rta_cd``.
"""

import time

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from model.analysis_function import AnalysisFunction
from model.linear_system import LinearSystem
from utils.exec_time import ExecTime

# Tuned analytic exponents for the starting assignment.
INIT_C = 1.0
INIT_A = 1.75
INIT_D = 1.0
# Multiplicative coordinate moves tried per task.
FACTORS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.45, 1.8, 2.3)
# Relaxed limit factor for the search objective.
LIMIT_FACTOR = 3.0
# Deterministic stop: number of full coordinate sweeps (a sweep tries every
# factor on every task). The descent also stops earlier when no factor improves.
MAX_SWEEPS = 16
# Anti-hang guard, in wall-clock seconds: abandons the sweep in progress. It is
# a safety valve, not the stopping rule. Runs report how often it is reached.
SAFETY_SECONDS = 1200.0


class RTACDAssignment(AnalysisFunction):
    def __init__(self, c=INIT_C, a=INIT_A, d=INIT_D, factors=FACTORS,
                 limit_factor=LIMIT_FACTOR, max_sweeps=MAX_SWEEPS,
                 safety_seconds=SAFETY_SECONDS):
        self.c = c
        self.a = a
        self.d = d
        self.factors = factors
        self.limit_factor = limit_factor
        self.max_sweeps = max_sweeps
        self.safety_seconds = safety_seconds
        self.sweeps = 0
        self.cap_hit = False
        self.timeout_hit = False
        self.exec_time = ExecTime()

    def analytic_init(self, system: LinearSystem) -> None:
        for flow in system.flows:
            ratio = flow.deadline / flow.period
            uflow = sum(t.wcet for t in flow.tasks) / flow.period
            for t in flow.tasks:
                t.deadline = (t.wcet ** self.c) * (ratio ** self.a) / (uflow ** self.d)

    def _sched(self, system: LinearSystem) -> bool:
        HolisticLocalEDFAnalysis(limit_factor=1, reset=True).apply(system)
        return system.is_schedulable()

    def _violation(self, system: LinearSystem) -> float:
        HolisticLocalEDFAnalysis(limit_factor=self.limit_factor, reset=False).apply(system)
        v = 0.0
        tot = 0.0
        for f in system.flows:
            r = f.wcrt
            x = 1e9 if r is None else r / f.deadline
            v += max(0.0, x - 1.0)
            tot += x if x < self.limit_factor else self.limit_factor
        return v + 1e-6 * tot

    def _descent(self, system: LinearSystem, t0: float) -> None:
        best = self._violation(system)
        deadline = t0 + self.safety_seconds
        self.sweeps = 0
        while True:
            if self.sweeps >= self.max_sweeps:
                self.cap_hit = True
                return
            self.sweeps += 1
            improved = False
            for task in system.tasks:
                cur = task.deadline
                for fac in self.factors:
                    if time.perf_counter() > deadline:
                        self.timeout_hit = True
                        task.deadline = cur
                        return
                    cand = cur * fac
                    if cand <= 1e-12:
                        continue
                    task.deadline = cand
                    o = self._violation(system)
                    if o < best - 1e-9:
                        best = o
                        cur = cand
                        improved = True
                task.deadline = cur
            if improved:
                if self._sched(system):
                    return
            else:
                return

    def apply(self, system: LinearSystem) -> LinearSystem:
        self.exec_time.init()
        self.sweeps = 0
        self.cap_hit = False
        self.timeout_hit = False
        t0 = time.perf_counter()
        self.analytic_init(system)
        if not self._sched(system):
            self._descent(system, t0)
        self.exec_time.stop()
        return system
