"""Work-and-position aware local-deadline assignment (WAD).

The rule derived in ``DERIVATION.md``::

    d_i = flow.deadline * ( Q * C_i / sum_C  +  S * i / (n - 1) ),

with ``Q = 1.6`` and ``S = 0.4`` for a flow whose tasks are ``i = 0 .. n-1``
and whose total worst-case execution time is ``sum_C = sum_j C_j``.

Two ingredients, one per term:

* ``Q * C_i / sum_C`` -- a *proportional* term: a task that executes more must
  be allowed more time, scaled up by a safety factor ``Q > 1`` because its
  response time exceeds its own execution (it also carries the interference of
  co-scheduled tasks).
* ``S * i / (n - 1)`` -- a *position* ramp: earlier tasks in the chain get
  smaller local deadlines (higher local-EDF priority), which keeps their
  response times -- and therefore the release jitter of their successors --
  small; the ramp hands the remaining end-to-end budget to the later tasks.

Both terms are needed: the proportional term alone (``S = 0``) collapses to a
uniform scale of PD and loses the chain structure; the ramp alone ignores the
execution-time spread.  See ``DERIVATION.md`` for the derivation and the
calibration of ``Q`` and ``S`` and for the failure cases.

``wad_lin`` is the closest sibling variant, using the *prior work* ``sum_{j<i}
C_j`` instead of the position ramp; it is registered for comparison.
"""

from algorithms import register

Q = 1.6
S = 0.4


def wad_shape(system, q=Q, s=S):
    """Assign the one-pass WAD deadlines in place (no analysis, no search)."""
    for flow in system.flows:
        tasks = flow.tasks
        n = len(tasks)
        total = sum(task.wcet for task in tasks)
        if total <= 0:
            total = 1.0
        for i, task in enumerate(tasks):
            pos = (i / (n - 1)) if n > 1 else 1.0
            task.deadline = flow.deadline * (q * task.wcet / total + s * pos)


@register
def wad(system):
    """Assign work-and-position aware local deadlines (see module docstring)."""
    wad_shape(system)


@register
def wad_lin(system):
    """Sibling variant: prior-work term instead of the position ramp.

    ``d_i = flow.deadline * (1.7 * C_i + 0.3 * sum_{j<i} C_j) / sum_C``.
    """
    for flow in system.flows:
        total = sum(task.wcet for task in flow.tasks)
        if total <= 0:
            total = 1.0
        prior = 0.0
        for task in flow.tasks:
            task.deadline = flow.deadline * (1.7 * task.wcet + 0.3 * prior) / total
            prior += task.wcet
