"""Back-loaded seed + bounded response-correction passes (the main rule).

Two stages (see ``DERIVATION.md`` §3):

1. **Seed** — assign the one-pass back-loaded, magnitude-scaled deadlines
   ``bl_scale`` (g = 0.3, λ = 2.0).
2. **Correction** — repeat ``P = 2`` times:
   - run the holistic analysis (with a large ``limit_factor`` so a flow's
     untruncated response is always available), then
   - for every flow scale **all** of its local deadlines by

         φ_f = clip( (D_f / R_f)^α , [lo, hi] ),   α = 0.6, [lo, hi] = [0.75, 1.25]

   where ``R_f`` is the flow's measured end-to-end response.

Rationale. The one-pass seed gets the *shape* right but its single global
magnitude cannot fit every flow: on the pool the seed is systematically too
generous for some flows and too tight for others. The correction treats each
flow separately: a flow whose response exceeds its deadline (``D_f < R_f``,
``φ_f < 1``) must shrink its local deadlines — that lowers its own response; a
flow with slack (``D_f > R_f``) can afford *larger* deadlines, which suppresses
the interference it imposes on other flows (eq 1). The exponent ``α < 1`` damps
the update and the clip keeps it in a **trust region**: (i) the response is not
linear in the deadlines, so an aggressive step overshoots, and (ii) shrinking a
flow's deadlines raises interference on its processor-mates, so a large downward
step has collateral cost. The corrected deadlines are *not* required to sum to
``D_f``.

This is a fixed two-pass rule (no trial-and-error search). It scores
912/1000 on the shipped pool (seed 42), versus 865 for the seed alone, 744 for
``pd_global`` and 708 for ``pd``; the coordinate-descent headroom is 945.
"""

from frozen.analysis import HolisticLocalEDFAnalysis

from algorithms import register
from algorithms.bl_scale import bl_shape_scale

# analysis used only to *read* responses; must never truncate at D_flow.
_BIG = 1e9
_PASSES = 2
_ALPHA = 0.6
_LO, _HI = 0.75, 1.25


def _clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


@register
def bl_adaptive(system):
    """Back-loaded seed (g=0.3, λ=2.0) + 2 bounded response-correction passes."""
    bl_shape_scale(system, g=0.3, lam=2.0)
    for _ in range(_PASSES):
        HolisticLocalEDFAnalysis(limit_factor=_BIG, reset=True).apply(system)
        for flow in system.flows:
            r = flow.wcrt
            if not r:
                continue
            phi = _clip((flow.deadline / r) ** _ALPHA, _LO, _HI)
            for task in flow.tasks:
                task.deadline *= phi
