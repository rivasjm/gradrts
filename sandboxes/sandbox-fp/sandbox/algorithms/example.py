"""Example algorithms -- a template, safe to edit or delete.

They show the contract: a callable that receives a system and assigns a
``priority`` to each task. **A larger priority number means a higher
priority.** Priorities only need to be ordered correctly; any monotone
rescaling is fine.
"""

from algorithms import register


@register
def by_task_order(system):
    """Assign increasing priorities in the order tasks appear in the system.

    Deliberately naive: it ignores deadlines and utilization entirely, so it
    is only useful to check that the plumbing works.
    """
    for i, task in enumerate(system.tasks):
        task.priority = i + 1


@register
def by_period(system):
    """Rate-monotonic-flavoured: shorter period -> higher priority.

    Priorities are assigned per processor. Note this is just a starting point;
    the deadlines are end-to-end and the analysis is holistic, so period alone
    is not necessarily a good signal.
    """
    for processor in system.processors:
        for rank, task in enumerate(sorted(processor.tasks, key=lambda t: t.period)):
            task.priority = rank + 1
