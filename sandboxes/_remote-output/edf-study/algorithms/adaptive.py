"""Load-adaptive variant (Q3).

Tries to pick the magnitude ``lambda`` from the measured mean processor
utilization instead of a single global constant::

    lambda(u) = max(0.5, 2.8 * u)

on top of the back-loaded shape g=0.3. On the seed-42 5x3x3 pool this scores
864/1000, statistically identical to the best *global* constant (g=0.3,
lambda=2.0, 865/1000). It is kept to document the negative Q3 result: on a pool
where every processor has the same utilization and every flow the same
structure, adaptivity buys nothing and mis-specified adaptivity hurts.
"""

from algorithms import register
from algorithms.shape import _shape_scale


@register
def pd_adaptive_u(system):
    """Back-loaded shape g=0.3 with lambda = max(0.5, 2.8 * system.utilization)."""
    lam = max(0.5, 2.8 * system.utilization)
    _shape_scale(system, g=0.3, lam=lam)
