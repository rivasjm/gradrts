from abc import ABC, abstractmethod

from model.linear_system import LinearSystem, Task


class AnalysisFunction(ABC):
    @abstractmethod
    def apply(self, system: LinearSystem) -> LinearSystem:
        pass

    def __call__(self, system: LinearSystem) -> LinearSystem:
        return self.apply(system)


def higher_priority(task: Task) -> list[Task]:
    return [t for t in task.processor.tasks
            if t.priority >= task.priority and t != task]


def init_wcrt(system: LinearSystem):
    for flow in system.flows:
        tasks = flow.tasks
        for i, task in enumerate(tasks):
            task.wcrt = task.wcet
            if i > 0:
                task.wcrt += tasks[i - 1].wcrt


def reset_wcrt(system: LinearSystem):
    for task in system.tasks:
        task.wcrt = None


def repr_wcrts(system: LinearSystem) -> str:
    msg = ""
    for flow in system.flows:
        ts = " ".join(map(lambda t: f"{t.wcrt if t.wcrt else -1:.2f}", flow.tasks))
        msg += f"{flow.period}: {ts} : {flow.deadline}\n"
    return msg


def debug_repr(system: LinearSystem):
    msg = ""
    for i, task in enumerate(system.tasks):
        msg += f"task {i} [proc={task.processor.name} prio={task.priority:.3f} C={task.wcet:.3f} T={task.flow.period:.3f} J={task.jitter:.3f}]\n"
    return msg


def calculate_priorities(system) -> bool:
    changed = False
    for processor in system.processors:
        tasks = sorted(processor.tasks,
                       key=lambda t: t.deadline,
                       reverse=True)
        for i, task in enumerate(tasks):
            if not changed and task.priority != i + 1:
                changed = True
            task.priority = i + 1
    return changed


def globalize_deadlines(system: LinearSystem):
    for flow in system.flows:
        tasks = flow.tasks
        if len(tasks) <= 1:
            continue
        for i, task in enumerate(tasks):
            if i == 0:
                continue
            task.deadline += tasks[i - 1].deadline


def clear_assignment(system):
    for t in system.tasks:
        t.priority = 1
        t.deadline = None


def normalize_priorities(system):
    max_priority = max(map(lambda t: t.priority, system.tasks))
    for t in system.tasks:
        t.priority = t.priority / max_priority


def extract_assignment(system: LinearSystem):
    tasks = system.tasks
    return [(t.priority, t.deadline, t.processor) for t in tasks]


def insert_assignment(system: LinearSystem, assignment):
    tasks = system.tasks
    for (prio, deadline, processor), task in zip(assignment, tasks):
        task.priority = prio
        task.deadline = deadline
        task.processor = processor


class LimitFactorReachedException(Exception):
    def __init__(self, task, response_time, limit):
        self.task = task
        self.response_time = response_time
        self.limit = limit
        self.message = f"Analysis stopped because provisional response time for task {task.name} (R={response_time}) " \
                       f"reached the limit (limit={limit})"
        super().__init__(self.message)
