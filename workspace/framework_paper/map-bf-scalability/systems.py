"""System pool for the map-bf scalability scenario.

The scenario studies how GDPA and the brute-force reference scale with the
number of tasks, at a *fixed* utilization (instead of sweeping the utilization
for a fixed number of tasks, as the other map-* scenarios do).

Population design
-----------------
Every system starts as a feasible 4-task base with layout ``(2, 2, 3)`` (2
flows x 2 tasks x 3 processors) and a contended initial mapping; the larger
sizes are obtained by successively appending one task at the end of a randomly
chosen flow, up to 16 tasks (layout ``(2, 8, 3)`` on average). This makes the
population *nested*: size ``n`` and ``n + 1`` share all their tasks and
mappings except the task appended going from ``n`` to ``n + 1``.

Why grow instead of shrink
--------------------------
Removing tasks from a 12-task base and rescaling to a fixed utilization
*amplifies* the utilization of the remaining tasks (the rescale factor is
``U*P / remaining_total > 1``), producing degenerate systems where a single
task needs more than one processor (about 90 % of the draws at n=4). Growing
does the opposite: appending a task raises the total and the rescale factor is
``U*P / (total + delta) < 1``, so every existing task only shrinks. The new
task is placed on the least-loaded processor, which empirically keeps every
processor below 1 at every size.

Feasibility
-----------
Each appended task gets a small pre-scaling utilization ``delta ~ U(0, DELTA_MAX)``
with period equal to its flow's period. Only the 4-task base has to be feasible:
a base that already has a processor at utilization >= 1 is discarded and another
one drawn (the UUNIFAST(n=4) feasibility rate); this only conditions on
feasibility and does not select across sizes.

Per size ``n`` the total utilization is fixed to ``U`` with
``set_system_utilization`` and each flow keeps one deadline factor drawn from
``U(DEADLINE_FACTOR_MIN, DEADLINE_FACTOR_MAX)``, so ``D = factor * F * T``
(``F`` = tasks currently in the flow, ``T`` = flow period) grows with the flow
and the deadline-per-step stays comparable across sizes. The default factor
range ``[0.5, 1.0]`` mirrors the other map-* scenarios and avoids the
degenerate, near-infeasible regime of a fixed ``0.5 * F * T``.

Running from ``code/``::

    .venv/bin/python workspace/framework_paper/map-bf-scalability/systems.py
"""

from copy import deepcopy
from random import Random
from typing import Dict, List, Optional

from examples.example_models import get_system
from examples.generator import set_system_utilization
from model.linear_system import LinearSystem, Task

# 4-task base: flows x tasks-per-flow x processors.
BASE_SIZE = (2, 2, 3)
# Total tasks of the systems evaluated; 16 is the grown target.
SIZES = tuple(range(4, 17))
N_SYSTEMS = 25
SEED = 42
UTILIZATION = 0.75
PERIOD_MIN = 100
PERIOD_MAX = 1000
# End-to-end deadline of a flow: D = factor * F * T, with one factor per flow
# drawn from U(DEADLINE_FACTOR_MIN, DEADLINE_FACTOR_MAX) and kept as it grows.
DEADLINE_FACTOR_MIN = 0.5
DEADLINE_FACTOR_MAX = 1.0
# Pre-scaling utilization of an appended task, drawn from U(0, DELTA_MAX).
# Must stay below 1 so that the rescaled task never needs a whole processor
# (see the module docstring); 0.5 is comfortably inside the safe range.
DELTA_MAX = 0.5
# Safety net so a pathological rejection rate fails loudly instead of hanging.
MAX_ATTEMPTS_FACTOR = 1000


def _new_base(rnd: Random, name: str, utilization: float,
              deadline_factor_min: float, deadline_factor_max: float) -> LinearSystem:
    """Generate a 4-task base with a contended mapping and per-flow deadlines."""
    system = get_system(BASE_SIZE, rnd, balanced=False, name=name,
                        deadline_factor_min=deadline_factor_min,
                        deadline_factor_max=deadline_factor_max,
                        period_min=PERIOD_MIN, period_max=PERIOD_MAX,
                        utilization=utilization)
    for flow in system.flows:
        factor = flow.deadline / (len(flow.tasks) * flow.period)
        flow.deadline = factor * len(flow.tasks) * flow.period
    return system


def over_utilized(system: LinearSystem) -> bool:
    """True when some processor reaches utilization >= 1."""
    return any(proc.utilization >= 1.0 for proc in system.processors)


def _grow(base: LinearSystem, rnd: Random,
          utilization: float) -> Optional[Dict[int, LinearSystem]]:
    """Derive every size from ``base`` by appending one task at a time to a
    random flow. Returns a ``{size: system}`` map, or ``None`` if some
    processor reaches utilization >= 1 at some size."""
    current = deepcopy(base)
    set_system_utilization(current, utilization)
    if over_utilized(current):
        return None

    by_size: Dict[int, LinearSystem] = {SIZES[0]: deepcopy(current)}
    for size in SIZES[1:]:
        flow = rnd.choice(current.flows)
        factor = flow.deadline / (len(flow.tasks) * flow.period)  # per-flow, kept
        task = Task(name=f"{flow.name}_t{size}",
                    wcet=rnd.uniform(0.0, DELTA_MAX) * flow.period)
        flow.add_tasks(task)
        flow.deadline = factor * len(flow.tasks) * flow.period
        task.processor = min(current.processors, key=lambda proc: proc.utilization)

        set_system_utilization(current, utilization)
        if over_utilized(current):
            return None
        by_size[size] = deepcopy(current)
    return by_size


def generate_pool(n_systems: int = N_SYSTEMS, seed: int = SEED,
                  utilization: float = UTILIZATION,
                  deadline_factor_min: float = DEADLINE_FACTOR_MIN,
                  deadline_factor_max: float = DEADLINE_FACTOR_MAX,
                  verbose: bool = True) -> List[List[LinearSystem]]:
    """Build the nested population as a ``systems[row][col]`` matrix.

    ``row`` is the base system (0..n_systems-1) and ``col`` indexes ``SIZES``,
    so ``systems[i][j]`` has ``SIZES[j]`` tasks and all systems in a column
    share that characteristic. The rows are nested (same ``i`` across columns
    is the same base) and independent (deep-copied), so a consumer may mutate
    them freely. Every system is named ``sys{i}_n{size}``.

    Each flow keeps one deadline factor drawn from
    ``U(deadline_factor_min, deadline_factor_max)``; the default (0.5, 0.5)
    gives the tightest ``D = 0.5 * F * T``.
    """
    rnd = Random(seed)
    matrix: List[List[LinearSystem]] = []
    attempts = 0
    rejected = 0
    max_attempts = max(n_systems * MAX_ATTEMPTS_FACTOR, MAX_ATTEMPTS_FACTOR)

    while len(matrix) < n_systems:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"giving up after {attempts} attempts: only "
                f"{len(matrix)}/{n_systems} systems accepted")
        index = len(matrix)
        base = _new_base(rnd, name=f"sys{index}", utilization=utilization,
                         deadline_factor_min=deadline_factor_min,
                         deadline_factor_max=deadline_factor_max)
        sizes = _grow(base, rnd, utilization)
        if sizes is None:
            rejected += 1
            continue
        row = []
        for size in SIZES:
            sizes[size].name = f"sys{index}_n{size}"
            row.append(sizes[size])
        matrix.append(row)

    if verbose:
        rate = rejected / attempts if attempts else 0.0
        print(f"pool: {n_systems} systems, sizes {SIZES[0]}..{SIZES[-1]}, "
              f"U={utilization}, deadline_factor={deadline_factor_min}..{deadline_factor_max}, "
              f"seed={seed}")
        print(f"attempts={attempts} rejected={rejected} ({rate:.1%} rejection rate)")
    return matrix


def by_size(matrix: List[List[LinearSystem]]) -> Dict[int, List[LinearSystem]]:
    """Column view of the matrix: ``{size: [systems]}`` (for diagnostics)."""
    return {size: [row[col] for row in matrix] for col, size in enumerate(SIZES)}


def summarize(matrix: List[List[LinearSystem]]) -> None:
    """Print per-size shape/utilization diagnostics of a generated pool."""
    print(f"{'size':>4} {'tasks':>6} {'flow_len':>9} {'avg_u':>8} "
          f"{'min_proc':>9} {'max_proc':>9} {'infeasible':>11}")
    for size, systems in by_size(matrix).items():
        proc_utils = [proc.utilization for s in systems for proc in s.processors]
        flow_lengths = [len(f.tasks) for s in systems for f in s.flows]
        infeasible = sum(over_utilized(s) for s in systems) / len(systems)
        print(f"{size:>4} "
              f"{sum(len(s.tasks) for s in systems) / len(systems):>6.1f} "
              f"{min(flow_lengths)}-{max(flow_lengths):<7} "
              f"{sum(s.utilization for s in systems) / len(systems):>8.3f} "
              f"{min(proc_utils):>9.3f} {max(proc_utils):>9.3f} "
              f"{infeasible:>11.1%}")


if __name__ == "__main__":
    summarize(generate_pool(verbose=True))
