from random import Random

import numpy as np

from examples.example_models import get_system
from examples.generator import set_utilization


def get_validation_example():
    rnd = Random(42)
    size = (3, 4, 3)  # flows, tasks, procs
    n = 50
    system = [get_system(size, rnd, balanced=True, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1) for i in range(n)][38]

    utilization = np.linspace(0.5, 0.9, 20)[13]
    set_utilization(system, utilization)
    return system