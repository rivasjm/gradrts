"""Diagnose the surrogate: parameter sweep on gradient alignment."""

import copy
import numpy as np
from random import Random

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import SequentialGradientFunction
from gradient_descent.parameter_handlers import DeadlineExtractor
from model.linear_system import SchedulerType
from surrogate.surrogate_edf import SurrogateEDFGradient


def cosine(fd, sr):
    fn, sn = np.linalg.norm(fd), np.linalg.norm(sr)
    if fn < 1e-9 or sn < 1e-9:
        return 0.0
    return np.dot(fd, sr) / (fn * sn)


def eval_config(systems, ph, cost_fn, **kw):
    cos_vals = []
    for sys_ in systems:
        s = copy.deepcopy(sys_)
        PDAssignment().apply(s)
        HolisticLocalEDFAnalysis(limit_factor=10, reset=False).apply(s)
        x = ph.extract(s)
        fd = np.array(SequentialGradientFunction(cost_function=cost_fn, sigma=1.5).compute(s, x))
        sr = np.array(SurrogateEDFGradient(**kw).compute(s, x))
        cos_vals.append(cosine(fd, sr))
    c = np.array(cos_vals)
    return np.mean(c), np.std(c), np.mean(c > 0)


def main():
    rnd = Random(42)
    systems = [get_system((3, 4, 3), rnd, balanced=True, name=str(i),
                          deadline_factor_min=0.3, sched=SchedulerType.EDF,
                          deadline_factor_max=0.7) for i in range(15)]

    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph,
                           analysis=HolisticLocalEDFAnalysis(limit_factor=10, reset=False))

    base = dict(tau=0.5, N_w=10, N_jitter=2, M_psi=50, temperature_max=0.1, grad_clip=1.0)

    configs = {
        "base":           base,
        "tau=0.1":        {**base, "tau": 0.1},
        "tau=1.0":        {**base, "tau": 1.0},
        "tau=0.05":       {**base, "tau": 0.05},
        "M_psi=100":      {**base, "M_psi": 100},
        "M_psi=200":      {**base, "M_psi": 200},
        "N_w=20":         {**base, "N_w": 20},
        "N_w=5":          {**base, "N_w": 5},
        "N_jitter=4":     {**base, "N_jitter": 4},
        "N_jitter=0":     {**base, "N_jitter": 0},
        "temp_max=0.01":  {**base, "temperature_max": 0.01},
        "temp_max=1.0":   {**base, "temperature_max": 1.0},
        "grad_clip=0.1":  {**base, "grad_clip": 0.1},
        "grad_clip=None": {**base, "grad_clip": None},
    }

    print(f"Systems: {len(systems)},  params: {len(configs)}")
    print(f"{'Config':<20s} {'cos(mean)':>8s} {'cos(std)':>8s} {'%>0':>6s}")
    print("-" * 50)

    best_cos, best_name = -1, ""
    for name, kw in configs.items():
        mc, sc, pc = eval_config(systems, ph, cost_fn, **kw)
        print(f"{name:<20s} {mc:8.4f} {sc:8.4f} {pc:6.1%}")
        if mc > best_cos:
            best_cos, best_name = mc, name

    print(f"\nBest: {best_name} (cos={best_cos:.4f})")

    # Test combined best
    best_kw = configs[best_name]
    combined_kw = {**best_kw, "M_psi": 200, "N_w": 20}
    mc, sc, pc = eval_config(systems, ph, cost_fn, **combined_kw)
    print(f"{'combined_best':<20s} {mc:8.4f} {sc:8.4f} {pc:6.1%}")


if __name__ == "__main__":
    main()
