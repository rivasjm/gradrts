"""WAD seed + bounded per-flow response-correction passes (the main rule).

Two stages (see ``DERIVATION.md``):

1. **Seed** -- the one-pass ``wad`` shape
   ``d_i = D_f * (Q * C_i / sum_C + S * i / (n-1))`` with ``Q = 1.6``,
   ``S = 0.4``.  It gets the *shape* right (proportional + back-loaded ramp) but
   spends one global magnitude for the whole system.

2. **Correction** -- repeat ``P = 3`` times:

   * analyse the system with a large ``limit_factor`` so every flow's
     untruncated end-to-end response ``R_f`` is available (never truncated at
     ``D_f``);
   * for each flow scale **all** its local deadlines by

     ``phi_f = clip( (D_f / R_f) ** alpha , [lo, hi] )``,

     with ``alpha = 0.5`` and ``[lo, hi] = [0.75, 1.25]``.

Rationale
---------
The seed's single global magnitude cannot fit every flow: some flows get more
deadline budget than they need, others too little.  A flow whose measured
response overshoots (``D_f < R_f`` => ``phi_f < 1``) shrinks its own deadlines,
which pulls its response down; a flow with slack (``D_f > R_f`` => ``phi_f > 1``)
can afford *larger* deadlines, which (eq. 1 of the analysis) suppresses the
interference it imposes on its co-processor tasks.  The exponent ``alpha < 1``
damps the step -- the response is not linear in the deadlines -- and the clip is
a trust region, because shrinking one flow's deadlines raises interference on
its processor-mates.  This is a **fixed count** of passes, not a search, and the
corrected deadlines are not required to sum to ``D_f``.

Scores 926/1000 on the shipped pool (seed 42) versus 875 for the one-pass seed,
744 for ``pd_global`` and 708 for ``pd``.  The one-pass rule ``wad`` lives in
``algorithms/wad.py``.
"""

from frozen.analysis import HolisticLocalEDFAnalysis

from algorithms import register
from algorithms.wad import wad_shape

# The analysis is used only to *read* responses; it must never truncate at D_f.
_BIG = 1e9
_PASSES = 3
_ALPHA = 0.5
_LO, _HI = 0.75, 1.25


def _clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


@register
def wad_adapt(system):
    """WAD seed + 3 bounded per-flow response-correction passes."""
    wad_shape(system)
    for _ in range(_PASSES):
        HolisticLocalEDFAnalysis(limit_factor=_BIG, reset=True).apply(system)
        for flow in system.flows:
            response = flow.wcrt
            if not response:
                continue
            phi = _clip((flow.deadline / response) ** _ALPHA, _LO, _HI)
            for task in flow.tasks:
                task.deadline *= phi
