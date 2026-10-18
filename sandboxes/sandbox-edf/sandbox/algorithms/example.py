"""Example algorithms -- a template, safe to edit or delete.

They show the contract: a callable that receives a system and assigns a
``deadline`` (local deadline) to each task. The flow's end-to-end deadline
(`task.flow.deadline`) is fixed; your local deadlines should normally not
exceed it in aggregate.
"""

from algorithms import register


@register
def equal_split(system):
    """Split each flow's end-to-end deadline equally among its tasks."""
    for flow in system.flows:
        n = len(flow.tasks)
        for task in flow.tasks:
            task.deadline = flow.deadline / n


@register
def by_wcet(system):
    """Proportional deadlines: local deadline proportional to each task's WCET."""
    for flow in system.flows:
        total = sum(task.wcet for task in flow.tasks)
        for task in flow.tasks:
            task.deadline = task.wcet * flow.deadline / total
