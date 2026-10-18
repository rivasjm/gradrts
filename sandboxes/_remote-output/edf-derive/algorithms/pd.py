"""Proportional-deadline (PD) assignment.

PD splits each flow's end-to-end deadline among its tasks in proportion to their
worst-case execution times::

    task.deadline = task.wcet * flow.deadline / sum(wcet over the flow's tasks)

The sum of a flow's local deadlines is then exactly the flow's end-to-end
deadline, so PD never over-spends a flow's budget. It is the baseline every new
technique is compared against.

This module ships two variants, both registered: ``pd`` (proportional local
deadlines) and ``pd_global`` (the same, made cumulative along each flow).
"""

from algorithms import register


@register
def pd(system):
    """Assign proportional-deadline (PD) local deadlines to every task."""
    for flow in system.flows:
        total = sum(task.wcet for task in flow.tasks)
        for task in flow.tasks:
            task.deadline = task.wcet * flow.deadline / total


@register
def pd_global(system):
    """PD local deadlines, accumulated along each flow (cumulative deadlines).

    Each task's deadline is the sum of its own proportional share and all its
    predecessors' shares, so the last task of a flow carries the whole end-to-end
    deadline::

        task[i].deadline = sum(task[j].wcet for j <= i) * flow.deadline / sum(wcet)
    """
    for flow in system.flows:
        total = sum(task.wcet for task in flow.tasks)
        cumulative = 0.0
        for task in flow.tasks:
            cumulative += task.wcet * flow.deadline / total
            task.deadline = cumulative
