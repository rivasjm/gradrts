from model.linear_system import LinearSystem


def backup_assignment(system: LinearSystem):
    tasks = system.tasks
    return [(t.priority, t.deadline, t.processor) for t in tasks]


def restore_assignment(system: LinearSystem, assignment):
    tasks = system.tasks
    for (prio, deadline, processor), task in zip(assignment, tasks):
        task.priority = prio
        task.deadline = deadline
        task.processor = processor