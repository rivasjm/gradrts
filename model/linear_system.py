import math
import sys
import warnings
from enum import Enum
from typing import List, Union, Callable

from model.system_model import SystemModel


class LinearSystem(SystemModel):
    """
    Base class representing a real-time system.

    This class serves as an interface for interacting with various
    real-time system implementations. Concrete subclasses should
    provide specific details about the system's tasks, processors,
    scheduling policy, and other relevant characteristics.
    """
    def __init__(self):
        """Initializes an system with a name, flows, and processors."""
        self.name = None
        self.flows: List[Flow] = []
        self.processors: List[Processor] = []

    def add_flows(self, *flows: "Flow") -> None:
        """
        Adds flows to the system.

        Args:
            *flows: Variable number of Flow objects to be added.
        """
        self.flows += flows
        for flow in flows:
            flow.system = self

    def add_procs(self, *procs: "Processor") -> None:
        """
        Adds processors to the system.

        Args:
            *procs: Variable number of Processor objects to be added.
        """
        for proc in procs:
            self.processors.append(proc)
            proc.system = self

    def __getitem__(self, item: Union[int, str, slice]) -> Union["Flow", List["Flow"], None]:
        """
        Allows accessing flows by index, name, or slice.

        Args:
            item: An integer index, a string name, or a slice object.

        Returns:
            The Flow object at the given index or with the given name,
            a list of Flow objects for a slice, or None if not found.
        """
        if isinstance(item, int) or isinstance(item, slice):
            return self.flows[item]
        elif isinstance(item, str):
            match = [flow for flow in self.flows if flow.name == item]
            return match[0] if match else None
        return None

    def apply(self, function: Callable) -> "LinearSystem":
        """
        Applies a given function to the system.

        Args:
            function: The function to be applied.

        Returns:
            The modified RTSystem object.
        """
        function(self)
        return self

    @property
    def tasks(self) -> List["Task"]:
        """Returns a list of all tasks in the system."""
        return [task for flow in self.flows for task in flow.tasks]

    def processor(self, name: str) -> Union["Processor", None]:
        """
        Returns the processor with the given name.

        Args:
            name: The name of the processor.

        Returns:
            The Processor object with the given name, or None if not found.
        """
        return next((p for p in self.processors if p.name == name), None)

    def is_schedulable(self) -> bool:
        """Returns True if all flows in the system are schedulable, False otherwise."""
        return all(flow.is_schedulable() for flow in self.flows)

    @property
    def utilization(self) -> float:
        """Returns the average utilization across all processors in the system."""
        utilizations = [proc.utilization for proc in self.processors]
        return sum(utilizations) / len(utilizations) if utilizations else 0

    @property
    def max_utilization(self) -> float:
        """Returns the maximum utilization among all processors in the system."""
        utilizations = [proc.utilization for proc in self.processors]
        return max(utilizations) if utilizations else 0

    @property
    def slack(self) -> float:
        """Returns the minimum slack among all flows in the system."""
        slacks = [flow.slack for flow in self.flows]
        return min(slacks) if slacks else sys.float_info.min

    @property
    def avg_flow_wcrt(self) -> float:
        """Returns the average worst-case response time across all flows in the system."""
        wcrts = [flow.wcrt for flow in self.flows]
        return sum(wcrts) / len(wcrts) if wcrts else 0

    @property
    def hyperperiod(self):
        return math.lcm(*[f.period for f in self.flows])

    def __repr__(self) -> str:
        """Returns a string representation of the system, listing its flows."""
        return "\n".join(str(flow) for flow in self.flows)


class SchedulerType(Enum):
    """Enum representing different scheduling policies."""
    FP = "Fixed_Priority"
    EDF = "EDF"


class Processor:
    """
    Represents a processor in the real-time system.

    Attributes:
        system: The RTSystem object this processor belongs to.
        name: The name of the processor.
        sched: The scheduling policy used by the processor (FP or EDF).
        local: True if the processor uses local clock synchronization (for EDF-L).
    """
    def __init__(self, name: str, sched: SchedulerType = SchedulerType.FP, local: bool = True):
        """
        Initializes a Processor with a name, scheduling policy, and locality flag.

        Args:
            name: The name of the processor.
            sched: The scheduling policy (FP or EDF). Defaults to FP.
            local: True for local clock synchronization (EDF-L). Defaults to True.
        """
        self.name: str = name
        self.sched: SchedulerType = sched
        self.local: bool = local
        self.system: LinearSystem = None

    def __repr__(self) -> str:
        """Returns a string representation of the processor (its name)."""
        return f"{self.name}"

    @property
    def tasks(self) -> List["Task"]:
        """Returns a list of tasks assigned to this processor."""
        return [task for flow in self.system.flows for task in flow.tasks
                if task.processor == self] if self.system else []

    @property
    def utilization(self) -> float:
        """Returns the total utilization of this processor."""
        utilizations = [task.wcet / task.period for task in self.tasks]
        return sum(utilizations)


class Flow:
    """
    Represents a flow of tasks in the real-time system.

    Attributes:
        system: The RTSystem object this flow belongs to.
        name: The name of the flow.
        period: The period of the flow.
        deadline: The deadline of the flow.
        tasks: A list of tasks in the flow.
        phase: The phase of the flow (for simulation).
        priority: An optional priority level for the flow (used in some scheduling policies).
    """
    def __init__(self, name: str, period: float, deadline: float, priority: int = None):
        """
        Initializes a Flow with a name, period, and deadline.

        Args:
            name: The name of the flow.
            period: The period of the flow.
            deadline: The deadline of the flow.
            priority: An optional priority level for the flow.
        """
        self.system: LinearSystem = None
        self.name: str = name
        self.period: float = period
        self.deadline: float = deadline
        self.tasks: List[Task] = []
        self.phase: float = 0
        self.priority: int = priority  # Added priority attribute

    def add_tasks(self, *tasks: "Task") -> None:
        """
        Adds tasks to the flow.

        Args:
            *tasks: Variable number of Task objects to be added.
        """
        self.tasks += tasks
        for task in tasks:
            task.flow = self

    def __repr__(self) -> str:
        """Returns a string representation of the flow, listing its tasks."""
        ts = " ".join(str(task) for task in self.tasks)
        return f"{self.period:.2f} : {ts} : {self.deadline:.2f})"

    @property
    def wcrt(self) -> Union[float, None]:
        """Returns the worst-case response time of the flow (from the last task)."""
        return self.tasks[-1].wcrt if self.tasks else None

    @property
    def slack(self) -> float:
        """Returns the slack of the flow."""
        if self.wcrt:
            return (self.deadline - self.wcrt) / self.deadline
        return float("-inf")

    def predecessors(self, task: "Task") -> List["Task"]:
        """
        Returns a list of predecessor tasks for the given task within the flow.

        Args:
            task: The task for which to find predecessors.

        Returns:
            A list of Task objects that are predecessors to the given task.
        """
        try:
            i = self.tasks.index(task)
            return [self.tasks[i-1]] if i > 0 else []
        except ValueError:
            raise ValueError(f"Task {task} not found in flow {self.name}")

    def successors(self, task: "Task") -> List["Task"]:
        """
        Returns a list of successor tasks for the given task within the flow.

        Args:
            task: The task for which to find successors.

        Returns:
            A list of Task objects that are successors to the given task.
        """
        try:
            i = self.tasks.index(task)
            return [self.tasks[i + 1]] if i < len(self.tasks) - 1 else []
        except ValueError:
            raise ValueError(f"Task {task} not found in flow {self.name}")

    def all_successors(self, task: "Task") -> List["Task"]:
        """
        Returns a list of all successor tasks for the given task within the flow.

        Args:
            task: The task for which to find all successors.

        Returns:
            A list of Task objects that are successors to the given task.
        """
        try:
            i = self.tasks.index(task)
            return self.tasks[i+1:]
        except ValueError:
            raise ValueError(f"Task {task} not found in flow {self.name}")

    def is_schedulable(self) -> bool:
        """Returns True if the flow is schedulable (wcrt <= deadline), False otherwise."""
        return self.wcrt is not None and self.wcrt <= self.deadline

    def __getitem__(self, item: Union[int, str, slice]) -> Union["Task", List["Task"], None]:
        """
        Allows accessing tasks by index, name, or slice.

        Args:
            item: An integer index, a string name, or a slice object.

        Returns:
            The Task object at the given index or with the given name,
            a list of Task objects for a slice, or None if not found.
        """
        if isinstance(item, int) or isinstance(item, slice):
            return self.tasks[item]
        elif isinstance(item, str):
            match = [task for task in self.tasks if task.name == item]
            return match[0] if match else None
        return None

class TaskType(Enum):
    """Enum representing different types of tasks."""
    ACTIVITY = "Activity"  # Represents a task that consumes processor time.
    OFFSET = "Offset"      # Represents a temporal offset within a flow.
    DELAY = "Delay"        # Represents a pure delay element within a flow.

class Task:
    """
    Represents a task in the real-time system.

    Attributes:
        flow: The Flow object this task belongs to.
        name: The name of the task.
        wcet: The worst-case execution time of the task.
        processor: The Processor object this task is assigned to.
        type: The type of the task (Activity, Offset, Delay).
        priority: The priority of the task (used in fixed-priority scheduling).
        deadline: The relative deadline of the task.
        wcrt: The worst-case response time of the task.
        bcet: The best-case execution time of the task.
    """
    def __init__(self,
                 name: str,
                 wcet: float,
                 processor: Processor = None,
                 type: TaskType = TaskType.ACTIVITY,
                 priority: int = 1,
                 bcet: float = 0,
                 deadline: float = 0):
        """
        Initializes a Task with the given parameters.

        Args:
            name: The name of the task.
            wcet: The worst-case execution time of the task.
            processor: The processor the task is assigned to.
            type: The type of the task (Activity, Offset, Delay). Defaults to Activity.
            priority: The priority of the task. Defaults to 1.
            bcet: The best-case execution time of the task. Defaults to 0.
            deadline: The relative deadline of the task. Defaults to 0.
        """
        self.name: str = name
        self.wcet: float = wcet
        self.processor: Processor = processor
        self.type: TaskType = type
        self.priority: int = priority
        self.deadline: float = deadline
        self.bcet: float = bcet
        self.wcrt: float = None
        self.flow: Flow = None

    def __repr__(self) -> str:
        """Returns a string representation of the task."""
        return f"{self.name} ({self.processor.name if self.processor else None},{self.utilization:.2f})"

    @property
    def utilization(self) -> float:
        """Returns the utilization of the task."""
        return self.wcet / self.period

    @property
    def period(self) -> float:
        """Returns the period of the task (inherited from its flow)."""
        return self.flow.period if self.flow else None

    @property
    def sched(self) -> SchedulerType:
        """Returns the scheduling policy used for this task (inherited from its processor)."""
        return self.processor.sched if self.processor else None

    @property
    def successors(self) -> List["Task"]:
        """Returns a list of successor tasks within the same flow."""
        return self.flow.successors(self) if self.flow else []

    @property
    def predecessors(self) -> List["Task"]:
        """Returns a list of predecessor tasks within the same flow."""
        return self.flow.predecessors(self) if self.flow else []

    @property
    def is_last(self) -> bool:
        """Returns True if this is the last task in its flow, False otherwise."""
        return len(self.successors) == 0

    @property
    def all_successors(self) -> List["Task"]:
        """Returns a list of all successor tasks within the same flow."""
        return self.flow.all_successors(self) if self.flow else []

    @property
    def jitter(self) -> float:
        """Returns the jitter of the task."""
        wcrts = [t.wcrt for t in self.predecessors]
        return max(wcrts) if wcrts else 0

    def copy(self) -> "Task":
        """Returns a copy of the task."""
        new_task = Task(name=self.name, wcet=self.wcet, processor=self.processor,
                        priority=self.priority, type=self.type, bcet=self.bcet,
                        deadline=self.deadline)
        new_task.wcrt = self.wcrt
        return new_task

def is_scheduler_type(system: LinearSystem, sched_type: SchedulerType) -> bool:
    """
    Checks if all processors in the system use the specified scheduler type.

    Args:
        system: The system to check.
        sched_type: The SchedulerType to compare against.

    Returns:
        True if all processors use the specified type, False otherwise.
    """
    return all(proc.sched == sched_type for proc in system.processors)

def save_attrs(elements: [], attrs: [str], key="_saved_") -> None:
    for element in elements:
        for attr in attrs:
            if hasattr(element, attr):
                value = getattr(element, attr)
                setattr(element, key + attr, value)
            else:
                warnings.warn(f"Warning, element {element} does not have attribute {attr}")


def restore_attrs(elements: [], attrs: [str], key="_saved_") -> None:
    for element in elements:
        for attr in attrs:
            if hasattr(element, key + attr):
                value = getattr(element, key + attr)
                setattr(element, attr, value)
