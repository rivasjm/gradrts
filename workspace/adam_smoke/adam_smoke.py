"""Compare the Adam implemented in the code with the standard Adam of Kingma &
Ba on the FP scenario with 15-task systems (brute force excluded).

Background
----------
``gradient_descent.update_functions.Adam`` updates the moments with
``(1 + beta)``::

    m = beta1 * m + (1 + beta1) * nabla
    v = beta2 * v + (1 + beta2) * nabla ** 2

whereas the recurrence described in the paper (and in the original Adam paper)
uses ``(1 - beta)``. This script:

1. **unit check** -- a corrected ``StandardAdam`` reproduces the standard
   recurrence and the current ``Adam`` deviates from it;
2. **end-to-end** -- runs GDPA with both variants (plus HOPA and PD when the full
   sweep is requested) on the FP 15-task systems.

It does not write anything and does not touch ``workspace/framework_paper``.

Run from ``code/``::

    # quick smoke test (10 systems, 3 levels, the two GDPA variants)
    .venv/bin/python workspace/adam_smoke/adam_smoke.py

    # full FP-15 sweep: 50 systems x 20 levels, GDPA (both) + HOPA + PD
    .venv/bin/python workspace/adam_smoke/adam_smoke.py --full
"""

import argparse
import copy
import math

import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from assignment.hopa_assignment import HOPAssignment
from examples.generator import set_utilization
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import FPHandler
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import Adam, NoisyAdam
from model.analysis_function import reset_wcrt
from vector.vector_fp import PrioritiesMatrix, VectorFPGradientFunction
from workspace.framework_paper.systems import get_systems

METHOD_ORDER = ("gdpa-current", "gdpa-standard", "hopa", "pd")


class StandardAdam(Adam):
    """Adam with the standard moment update, ``(1 - beta)``."""

    def update(self, x, nabla, t):
        if not self.size:
            self.size = len(nabla)
            self.m = [0.0] * self.size
            self.v = [0.0] * self.size

        updates = [0.0] * self.size
        for i in range(self.size):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * nabla[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * nabla[i] ** 2

            me = self.m[i] / (1 - self.beta1 ** t)
            ve = self.v[i] / (1 - self.beta2 ** t)

            updates[i] = -self.lr * me / (math.sqrt(ve) + self.epsilon)
        return updates


class StandardNoisyAdam(NoisyAdam):
    """``NoisyAdam`` whose Adam sub-component uses ``StandardAdam``."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.adam = StandardAdam(lr=self.adam.lr, beta1=self.adam.beta1,
                                 beta2=self.adam.beta2, epsilon=self.adam.epsilon)


def unit_check(steps=50, gradient=1.0, lr=3.0, epsilon=0.1):
    """Compare one update of the current and the corrected Adam for a constant
    gradient and return ``(current_update, standard_update, expected_standard)``.

    With a constant gradient the bias correction cancels the history, so the
    standard update has a closed form: ``-lr * g / (|g| + epsilon)``.
    """
    current = Adam(lr=lr, epsilon=epsilon)
    standard = StandardAdam(lr=lr, epsilon=epsilon)
    for t in range(1, steps + 1):
        u_current = current.update([0.0], [gradient], t)[0]
        u_standard = standard.update([0.0], [gradient], t)[0]
    expected = -lr * gradient / (abs(gradient) + epsilon)
    return u_current, u_standard, expected


def run_gdpa(system, update_function, limit):
    """GDPA over fixed priorities, as in ``fp/fp.py`` (PD start, vector gradient)."""
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = FPHandler()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=limit)
    gradient_function = VectorFPGradientFunction(PrioritiesMatrix())
    optimizer = GradientDescentOptimizer(parameter_handler=parameter_handler,
                                         cost_function=cost_function,
                                         stop_function=stop_function,
                                         gradient_function=gradient_function,
                                         update_function=update_function,
                                         verbose=False)
    reset_wcrt(system)
    PDAssignment(normalize=True).apply(system)
    optimizer.apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def run_hopa(system):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    reset_wcrt(system)
    HOPAssignment(analysis=analysis).apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def run_pd(system):
    reset_wcrt(system)
    PDAssignment(normalize=True).apply(system)
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    return system.is_schedulable()


def build_methods(names, limit, seed):
    """Return an ordered ``{name: callable(system) -> bool}`` mapping."""
    runners = {
        "gdpa-current": lambda s: run_gdpa(s, NoisyAdam(seed=seed), limit),
        "gdpa-standard": lambda s: run_gdpa(s, StandardNoisyAdam(seed=seed), limit),
        "hopa": run_hopa,
        "pd": run_pd,
    }
    return {name: runners[name] for name in names}


def evaluate(systems, utilizations, methods):
    """Run every method on every (system, utilization); return ``{name: bool[U][N]}``."""
    results = {name: np.zeros((len(utilizations), len(systems)), dtype=bool)
               for name in methods}
    for ui, u in enumerate(utilizations):
        for si, base in enumerate(systems):
            set_utilization(base, u)
            for name, func in methods.items():
                system = copy.deepcopy(base)
                results[name][ui, si] = func(system)
    return results


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="full FP-15 sweep: 50 systems x 20 levels and all methods")
    parser.add_argument("--systems", type=int, default=None,
                        help="number of FP 15-task systems (default: 10, or 50 with --full)")
    parser.add_argument("--utilizations", type=float, nargs="+", default=None,
                        help="utilization levels (default: 0.7 0.8 0.9, or the "
                             "20 FP-15 levels with --full)")
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=None,
                        help="methods to run (default: the two GDPA variants, or "
                             "all of them with --full)")
    parser.add_argument("--limit", type=int, default=100,
                        help="GDPA iteration budget (default: 100)")
    parser.add_argument("--seed", type=int, default=1,
                        help="noise seed, shared by both GDPA variants (default: 1)")
    args = parser.parse_args()

    print("=== 1. Unit check: current Adam vs standard Adam ===")
    u_current, u_standard, expected = unit_check()
    print(f"  current  Adam update = {u_current:+.6f}")
    print(f"  standard Adam update = {u_standard:+.6f}  (analytic {expected:+.6f})")
    print(f"  ratio current/standard = {u_current / u_standard:.4f}")
    assert abs(u_standard - expected) < 1e-9, "StandardAdam does not match standard Adam"
    assert abs(u_current - u_standard) > 0.1, "current Adam matches standard Adam"
    print("  OK: the corrected Adam is standard, the current one is not.\n")

    n_systems = args.systems if args.systems is not None else (50 if args.full else 10)
    if args.utilizations is not None:
        utilizations = list(args.utilizations)
    elif args.full:
        utilizations = list(np.linspace(0.5, 0.9, 20))
    else:
        utilizations = [0.7, 0.8, 0.9]
    names = args.methods or (["gdpa-current", "gdpa-standard"] if not args.full
                             else list(METHOD_ORDER))

    print("=== 2. End-to-end: GDPA (both Adams) / HOPA / PD on FP 15-task systems ===")
    systems = get_systems(15)[:n_systems]
    print(f"  systems={len(systems)}  levels={len(utilizations)}  "
          f"methods={names}  limit={args.limit}  seed={args.seed}")
    methods = build_methods(names, args.limit, args.seed)
    results = evaluate(systems, utilizations, methods)

    total = len(systems) * len(utilizations)
    print(f"\n  schedulable systems (out of {total})")
    for name in names:
        print(f"    {name:14s}: {int(results[name].sum())}")

    print("\n  per utilization level")
    header = "    u     " + "".join(f"{n:>15s}" for n in names)
    print(header)
    for ui, u in enumerate(utilizations):
        row = "".join(f"{int(results[n][ui].sum()):>15d}" for n in names)
        print(f"    {u:.3f} {row}")

    if "gdpa-current" in names and "gdpa-standard" in names:
        diff = int((results["gdpa-current"] != results["gdpa-standard"]).sum())
        print(f"\n  cases where the two GDPA variants differ: {diff} "
              f"({100.0 * diff / total:.1f}%)")


if __name__ == "__main__":
    main()
