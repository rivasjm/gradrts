"""Back-loaded, magnitude-scaled local-deadline assignment (Q1/Q2 findings).

Two closed forms discovered in the study, both registered:

``pd_cum_scaled``  unconstrained best of the 2-D (shape, magnitude) family:
    d_i = (share_i + g * prefix_{i-1}) * scale,  scale = lam * D_flow / sum(...)
    with g=0.3, lam=2.0. It overspends each flow's budget
    (sum(d_i)/D_flow ~ 1.3-1.6) and scores 865/1000 on the seed-42 pool.

``pd_budget_cum``  the best *budget-feasible* variant (sum(d_i) == D_flow):
    the same cumulative shape at g=0.6, lam=1.0, scoring 762/1000.

Rationale (see NOTES.md): the analysis credits an interfering task with fewer
activations the later its local deadline (eq 1), while the analysed task's own
deadline largely cancels in eq (9). Back-loading (later tasks bigger) plus a
uniform magnitude inflation therefore reduces counted interference. The first
form spends more than the flow budget; the second does not.
"""

from algorithms import register


def _shape_scale(system, g, lam):
    for flow in system.flows:
        total = sum(task.wcet for task in flow.tasks)
        ds, prev = [], 0.0
        for task in flow.tasks:
            s = task.wcet * flow.deadline / total
            ds.append(s + g * prev)
            prev += s
        scale = lam * flow.deadline / sum(ds)
        for task, d in zip(flow.tasks, ds):
            task.deadline = d * scale


@register
def pd_cum_scaled(system):
    """Back-loaded cumulative shape (g=0.3) scaled by lambda=2.0 (overspends)."""
    _shape_scale(system, g=0.3, lam=2.0)


@register
def pd_budget_cum(system):
    """Budget-feasible back-loaded cumulative shape (g=0.6), sum(d_i)==D_flow."""
    _shape_scale(system, g=0.6, lam=1.0)
