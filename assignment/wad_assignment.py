"""Local-EDF deadline assignment by the work-and-position aware rule (WAD).

A one-pass closed form derived from the holistic local-EDF analysis (see the
``sandbox-edf-derive`` study). For a flow with tasks ``i = 0 .. n-1``, total
worst-case execution time ``sum_C`` and end-to-end deadline ``D``::

    d_i = D * ( Q * C_i / sum_C  +  S * i / (n - 1) ),     Q = 1.6, S = 0.4

Two ingredients, one per term:

* ``Q * C_i / sum_C`` -- a *proportional* term: a task that executes more is
  allowed more time, scaled up by a safety factor ``Q > 1`` because its response
  time exceeds its own execution (it also carries the interference of
  co-scheduled tasks).
* ``S * i / (n - 1)`` -- a *position* ramp: earlier tasks in the chain get
  smaller local deadlines (higher local-EDF priority), which keeps their response
  times -- and therefore the release jitter of their successors -- small; the
  ramp hands the remaining end-to-end budget to the later tasks.

Both terms are needed: the proportional term alone (``S = 0``) collapses to a
uniform scale of PD and loses the chain structure; the ramp alone ignores the
execution-time spread.

No analysis is run and no search: the assignment is a single pass over the flows.
Scores 875/1000 on the reference pool (seed 42, 5x3x3), versus 708 for ``pd`` and
744 for ``pd_global``; the correction-pass variant ``WADAdaptAssignment``
(``wad_adapt_assignment.py``) scores ~930. The rule deliberately overspends the
per-flow budget (``sum_i d_i > D``); that is a property of the approximate
analysis used here.

Ported from the sandbox agent's ``wad``.
"""

from model.analysis_function import AnalysisFunction
from model.linear_system import LinearSystem
from utils.exec_time import ExecTime

# Calibrated constants: proportional factor and position-ramp amplitude.
Q = 1.6
S = 0.4


class WADAssignment(AnalysisFunction):
    def __init__(self, q=Q, s=S):
        self.q = q
        self.s = s
        self.exec_time = ExecTime()

    @staticmethod
    def calculate_local_deadlines(system: LinearSystem, q=Q, s=S) -> None:
        """Assign the one-pass WAD local deadlines in place (no analysis, no search)."""
        for flow in system.flows:
            tasks = flow.tasks
            n = len(tasks)
            total = sum(task.wcet for task in tasks)
            if total <= 0:
                total = 1.0
            for i, task in enumerate(tasks):
                pos = (i / (n - 1)) if n > 1 else 1.0
                task.deadline = flow.deadline * (q * task.wcet / total + s * pos)

    def apply(self, system: LinearSystem) -> LinearSystem:
        self.exec_time.init()
        self.calculate_local_deadlines(system, self.q, self.s)
        self.exec_time.stop()
        return system
