"""Shared population of synthetic systems for the paper evaluations.

Task utilizations are drawn globally (see ``examples.generator.generate_system``)
so that the balanced and unbalanced populations share exactly the same task set;
only the initial task-to-processor mapping differs. Every scenario starts from the
balanced population except the MAP unbalanced variant.

``SIZES`` maps the size key (total number of tasks) to the ``(flows, tasks,
processors)`` tuple passed to the generator. The population is generated with a
fixed seed so that repeated runs are reproducible.
"""

from random import Random

from examples.example_models import get_system

SIZES = {
    15: (5, 3, 3),
    25: (5, 5, 5),
}

SEED = 42
POPULATION = 50
DEADLINE_FACTOR_MIN = 0.5
DEADLINE_FACTOR_MAX = 1
PERIOD_MIN = 100
PERIOD_MAX = 1000


def _get_systems(size, balanced):
    """Generate the population for ``size`` with the requested mapping.

    A fresh population is returned on every call so that callers can mutate the
    systems (set utilization, change the scheduler or the mapping) without
    affecting other callers. The task set (periods, deadlines and utilizations)
    is identical for ``balanced=True`` and ``balanced=False``; only the initial
    task-to-processor mapping differs.
    """
    if size not in SIZES:
        raise ValueError(f"unknown system size {size!r}, use one of {sorted(SIZES)}")

    rnd = Random(SEED)
    return [get_system(SIZES[size], rnd, balanced=balanced, name=str(i),
                       deadline_factor_min=DEADLINE_FACTOR_MIN,
                       deadline_factor_max=DEADLINE_FACTOR_MAX,
                       period_min=PERIOD_MIN, period_max=PERIOD_MAX)
            for i in range(POPULATION)]


def get_systems_15():
    return _get_systems(15, balanced=True)


def get_systems_25():
    return _get_systems(25, balanced=True)


def get_systems_15_unbalanced():
    return _get_systems(15, balanced=False)


def get_systems_25_unbalanced():
    return _get_systems(25, balanced=False)


def get_systems(size, balanced=True):
    return _get_systems(size, balanced)
