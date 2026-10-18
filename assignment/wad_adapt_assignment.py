"""Local-EDF deadline assignment: WAD seed + bounded per-flow correction passes.

Two stages (see the ``sandbox-edf-derive`` study):

1. **Seed** -- the one-pass WAD shape
   (``WADAssignment.calculate_local_deadlines``):
   ``d_i = D * (Q * C_i / sum_C + S * i / (n - 1))`` with ``Q = 1.6``,
   ``S = 0.4``. It gets the *shape* right (proportional + back-loaded ramp) but
   spends one global magnitude for the whole system.

2. **Correction** -- repeat ``P = 3`` times:

   * analyse the system with a huge ``limit_factor`` so every flow's untruncated
     end-to-end response ``R_f`` is available (never truncated at ``D_f``);
   * for each flow scale **all** its local deadlines by

     ``phi_f = clip( (D_f / R_f) ** alpha , [lo, hi] )``,

     with ``alpha = 0.5`` and ``[lo, hi] = [0.75, 1.25]``.

   A flow whose response overshoots (``D_f < R_f`` -> ``phi_f < 1``) shrinks its
   deadlines, pulling its response down; a flow with slack (``D_f > R_f`` ->
   ``phi_f > 1``) can afford *larger* deadlines, which (via the analysis' local
   deadline cap) suppresses the interference it imposes on its co-processor
   tasks. The exponent ``alpha < 1`` damps the step, and the clip is a trust
   region, because shrinking one flow's deadlines raises interference on its
   processor-mates.

This is a **fixed count** of passes, not a search. The corrected deadlines are not
required to sum to ``D_f`` -- the rule deliberately overspends the per-flow
budget, a property of the approximate analysis.

Scores ~930/1000 on the reference pool (seed 42, 5x3x3), versus 875 for the
one-pass ``WADAssignment``, 744 for ``pd_global`` and 708 for ``pd``.

Ported from the sandbox agent's ``wad_adapt``.
"""

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.wad_assignment import Q as SEED_Q, S as SEED_S, WADAssignment
from model.analysis_function import AnalysisFunction
from model.linear_system import LinearSystem
from utils.exec_time import ExecTime

# The analysis is used only to *read* responses; it must never truncate at D_f.
LIMIT_FACTOR = 1e9
# Wall-clock cap per analysis (s): guards against pathological systems where an
# untruncated fixed point explodes. On timeout the analysis aborts and the flow
# is simply left uncorrected in that pass.
MAX_TIME = 2.0
# Fixed number of correction passes.
PASSES = 3
# Damping exponent and trust region for the per-flow scaling.
ALPHA = 0.5
LO = 0.75
HI = 1.25


def _clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


class WADAdaptAssignment(AnalysisFunction):
    def __init__(self, passes=PASSES, alpha=ALPHA, lo=LO, hi=HI,
                 limit_factor=LIMIT_FACTOR, budget=False, max_time=MAX_TIME,
                 q=SEED_Q, s=SEED_S):
        self.passes = passes
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.limit_factor = limit_factor
        self.budget = budget
        self.max_time = max_time
        self.q = q
        self.s = s
        self.exec_time = ExecTime()

    @staticmethod
    def _project(system: LinearSystem) -> None:
        """Rescale each flow so that ``sum_i d_i == D_f`` (budget-feasible)."""
        for flow in system.flows:
            total = sum(task.deadline for task in flow.tasks)
            if total <= 0:
                continue
            k = flow.deadline / total
            for task in flow.tasks:
                task.deadline *= k

    def apply(self, system: LinearSystem) -> LinearSystem:
        self.exec_time.init()
        WADAssignment.calculate_local_deadlines(system, self.q, self.s)
        if self.budget:
            self._project(system)
        for _ in range(self.passes):
            HolisticLocalEDFAnalysis(limit_factor=self.limit_factor, reset=True,
                                     max_time=self.max_time).apply(system)
            for flow in system.flows:
                response = flow.wcrt
                if not response:
                    continue
                phi = _clip((flow.deadline / response) ** self.alpha, self.lo, self.hi)
                for task in flow.tasks:
                    task.deadline *= phi
            if self.budget:
                self._project(system)
        self.exec_time.stop()
        return system
