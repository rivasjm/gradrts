"""Frozen evaluation core for the priority-assignment sandbox.

Import from here (read-only):

    from frozen import HolisticFPAnalysis, get_systems, set_utilization
    from frozen import UTILIZATIONS, SEED, POOLS, REFERENCE_METHOD

Nothing in this package may be modified: it is the fixed referee used to score
every priority-assignment algorithm you write in ``algorithms/``.
"""

from .model import (Flow, LinearSystem, Processor, SchedulerType, Task,
                    TaskType, higher_priority)
from .analysis import HolisticFPAnalysis
from . import pool
from . import baselines
from . import harness
from . import results
from .pool import (POPULATION, POOLS, REFERENCE_METHOD, SEED, SIZES,
                   UTILIZATIONS, get_systems, set_utilization)
from .harness import evaluate, is_schedulable

__all__ = [
    "Flow", "LinearSystem", "Processor", "SchedulerType", "Task", "TaskType",
    "higher_priority", "HolisticFPAnalysis", "pool", "baselines", "harness",
    "results",
    "get_systems", "set_utilization", "UTILIZATIONS", "SEED", "POOLS",
    "REFERENCE_METHOD", "POPULATION", "SIZES", "evaluate", "is_schedulable",
]
