"""Shared population sweep, matching ``SchedRatioEval``'s convention.

The runner keeps one population and calls ``set_utilization`` on the same system
objects at every utilization level. Mathematically each level is the same task
set, but re-scaling in place accumulates a few ulps of rounding in the WCETs.
The holistic analysis is discontinuous at a handful of instances, so those ulps
can flip a verdict. Auxiliary scripts must use this same convention or their
counts will not match the main tables (e.g. PD 710 versus 706).
"""
from examples.generator import set_utilization, to_edf
from workspace.framework_paper.systems import get_systems


def base_population(size):
    """Generate the benchmark population and switch it to local EDF."""
    systems = get_systems(size)
    for s in systems:
        to_edf(s)
    return systems


def levels(systems, utilizations, record=None):
    """Yield ``(index, utilization, systems)``, scaling in place at each level.

    ``record`` is an optional predicate on the level index; when it returns
    false the level is still scaled (to keep the same rounding sequence) but not
    yielded.
    """
    for i, u in enumerate(utilizations):
        for s in systems:
            set_utilization(s, u)
        if record is None or record(i):
            yield i, u, systems
