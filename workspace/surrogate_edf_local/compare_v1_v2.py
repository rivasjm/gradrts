"""Compare v1 (grid) vs v2 (real psi set) surrogate gradient alignment."""

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
from workspace.surrogate_edf_local.edf_gradient_v2 import SurrogateEDFGradientV2


def cosine(fd, sr):
    fn, sn = np.linalg.norm(fd), np.linalg.norm(sr)
    if fn < 1e-9 or sn < 1e-9:
        return 0.0
    return np.dot(fd, sr) / (fn * sn)


def main():
    rnd = Random(42)
    n_sys = 15
    systems = [
        get_system((3, 4, 3), rnd, balanced=True, name=str(i),
                   deadline_factor_min=0.3, sched=SchedulerType.EDF,
                   deadline_factor_max=0.7)
        for i in range(n_sys)
    ]

    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph,
                           analysis=HolisticLocalEDFAnalysis(limit_factor=10, reset=False))

    v1_kw = dict(tau=0.1, N_w=10, N_jitter=2, M_psi=50, temperature_max=0.1, grad_clip=1.0)
    v2_kw = dict(tau=0.1, N_w=10, N_jitter=2, temperature_max=0.1, grad_clip=1.0)

    print(f"{'sys':>4s} {'cos_v1':>8s} {'cos_v2':>8s} {'delta':>8s}")
    print("-" * 35)

    cos_v1_list, cos_v2_list = [], []

    for i, sys_ in enumerate(systems):
        s = copy.deepcopy(sys_)
        PDAssignment().apply(s)
        HolisticLocalEDFAnalysis(limit_factor=10, reset=False).apply(s)
        x = ph.extract(s)

        fd = np.array(SequentialGradientFunction(cost_function=cost_fn, sigma=1.5).compute(s, x))
        v1 = np.array(SurrogateEDFGradient(**v1_kw).compute(s, x))
        v2 = np.array(SurrogateEDFGradientV2(**v2_kw).compute(s, x))

        c1 = cosine(fd, v1)
        c2 = cosine(fd, v2)
        cos_v1_list.append(c1)
        cos_v2_list.append(c2)
        print(f"{i:4d} {c1:8.4f} {c2:8.4f} {c2-c1:+8.4f}")

    cos_v1 = np.array(cos_v1_list)
    cos_v2 = np.array(cos_v2_list)

    print(f"\n{'Summary':>30s}")
    print(f"  V1 (grid):    mean cos={np.mean(cos_v1):.4f}  std={np.std(cos_v1):.4f}  >0={np.mean(cos_v1>0):.1%}")
    print(f"  V2 (psi set): mean cos={np.mean(cos_v2):.4f}  std={np.std(cos_v2):.4f}  >0={np.mean(cos_v2>0):.1%}")
    print(f"  V2 better in {np.sum(cos_v2 > cos_v1)}/{n_sys} systems")
    print(f"  Mean delta: {np.mean(cos_v2 - cos_v1):+.4f}")


if __name__ == "__main__":
    main()
