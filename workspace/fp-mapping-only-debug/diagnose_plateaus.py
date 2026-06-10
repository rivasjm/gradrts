"""
Diagnose plateauing behaviour in the mapping-only gradient optimizer.

Runs a single system at a single utilization with verbose per-iteration logging
to show whether the optimizer is making progress or is stuck in flat regions of
the loss landscape.
"""
import numpy as np
import time
from random import Random

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from examples.generator import set_utilization
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import MappingOnlyExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from vector.vector_fp import VectorFPGradientFunction, MappingOnlyMatrix


class PlateauDiagnostic:
    """Callback that tracks and reports optimization progress per iteration."""

    def __init__(self, n_tasks: int, n_procs: int):
        self.n_tasks = n_tasks
        self.n_procs = n_procs
        self.prev_discrete = None           # previous iteration's discrete mapping
        self.mapping_flips = None           # per-task: True if mapping changed this iteration
        self.improved = None                # cost < previous best?
        self.total_improvements = 0
        self.total_zero_grad = 0            # iterations with all-zero gradient (via update norm)
        self.iteration_data = []

    @staticmethod
    def discrete_mapping(x, n_tasks, n_procs):
        """Extract the discrete mapping from continuous parameters."""
        mapping = []
        for t in range(n_tasks):
            sub = x[t * n_procs: (t + 1) * n_procs]
            mapping.append(max(range(len(sub)), key=lambda i: sub[i]))
        return mapping

    def __call__(self, t, S, x, xb, cost, best, ref_cost):
        discrete = self.discrete_mapping(x, self.n_tasks, self.n_procs)

        if self.prev_discrete is not None:
            self.mapping_flips = [d != p for d, p in zip(discrete, self.prev_discrete)]
            flips = sum(self.mapping_flips)
        else:
            flips = 0

        self.improved = cost < best

        # Count occurrences of each mapping across tasks
        proc_counts = [0] * self.n_procs
        for d in discrete:
            proc_counts[d] += 1

        self.iteration_data.append({
            't': t,
            'cost': cost,
            'best': best,
            'flips': flips,
            'improved': self.improved,
            'proc_counts': tuple(proc_counts),
        })

        if self.improved:
            self.total_improvements += 1

        # Format per-iteration line
        improved_mark = " *" if self.improved else "  "
        proc_str = "/".join(str(c) for c in proc_counts)
        print(f"  iter {t:3d}{improved_mark} cost={cost:8.4f} best={best:8.4f}  "
              f"flips={flips:2d}  procs=[{proc_str}]")

        self.prev_discrete = discrete[:]


def run_diagnostic(system_name="diag", utilization=0.6, seed=42,
                   sigma=3.0, patience=20, limit=100, verbose=True):
    """
    Run the mapping-only optimiser on one system with full diagnostics.

    Parameters
    ----------
    system_name : str
        Label for the generated system.
    utilization : float
        Target average utilisation.
    seed : int
        Random seed for system generation.
    sigma : float
        Perturbation spread for finite-difference gradients.
    patience : int
        Stop after this many iterations without cost improvement.
    limit : int
        Hard iteration cap.
    verbose : bool
        Print per-iteration details.
    """
    rnd = Random(seed)
    size = (3, 4, 3)  # flows, tasks, procs
    system = get_system(size, rnd, balanced=False, name=system_name,
                        deadline_factor_min=0.5, deadline_factor_max=1)
    set_utilization(system, utilization)

    n_tasks = len(system.tasks)
    n_procs = len(system.processors)

    print(f"System '{system_name}': {n_tasks} tasks, {n_procs} procs, u={utilization}")
    print(f"Parameters: sigma={sigma}, patience={patience}, limit={limit}")
    print(f"Gradient parameter count: {n_tasks * n_procs}")
    print()

    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = MappingOnlyExtractor()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=limit, patience=patience)
    gradient_function = VectorFPGradientFunction(
        scenarios_builder=MappingOnlyMatrix(), sigma=sigma, cost_limit_factor=3)

    update_function = NoisyAdam(lr=1.5, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.5)

    diagnostic = PlateauDiagnostic(n_tasks, n_procs) if verbose else None

    optimizer = GradientDescentOptimizer(
        parameter_handler=parameter_handler,
        cost_function=cost_function,
        stop_function=stop_function,
        gradient_function=gradient_function,
        update_function=update_function,
        callback=diagnostic,
        verbose=False)

    PDAssignment(normalize=True).apply(system)

    t0 = time.perf_counter()
    optimizer.apply(system)
    elapsed = time.perf_counter() - t0

    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    schedulable = system.is_schedulable()

    print()
    print("─" * 60)
    print(f"Result: {'SCHEDULABLE' if schedulable else 'UNSCHEDULABLE'}")
    print(f"Total iterations: {len(diagnostic.iteration_data)}")
    print(f"Iterations with improvement: {diagnostic.total_improvements}")
    print(f"Improvement ratio: {diagnostic.total_improvements / len(diagnostic.iteration_data):.1%}")
    print(f"Final mapping (task -> proc): {diagnostic.prev_discrete}")
    print(f"Final best cost: {stop_function.solution_cost():.4f}")
    print(f"Elapsed: {elapsed:.2f}s")

    # Plateau analysis
    plateau_runs = []
    current_run = 0
    for entry in diagnostic.iteration_data:
        if entry['improved']:
            if current_run > 0:
                plateau_runs.append(current_run)
            current_run = 0
        else:
            current_run += 1
    if current_run > 0:
        plateau_runs.append(current_run)

    if plateau_runs:
        print(f"Plateau lengths: min={min(plateau_runs)}, max={max(plateau_runs)}, "
              f"mean={np.mean(plateau_runs):.1f}, median={np.median(plateau_runs):.1f}")
    else:
        print("No plateaus detected (improved every iteration)")

    if not schedulable:
        print("NOTE: system unschedulable — plateau behaviour is expected as cost bottoms out")

    return system, diagnostic.iteration_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Diagnose plateau behaviour in mapping-only gradient optimisation")
    parser.add_argument("-u", "--utilization", type=float, default=0.6,
                        help="Target utilisation (default: 0.6)")
    parser.add_argument("-s", "--sigma", type=float, default=3.0,
                        help="Perturbation spread sigma (default: 3.0)")
    parser.add_argument("-p", "--patience", type=int, default=20,
                        help="Patience for early stopping (default: 20)")
    parser.add_argument("-l", "--limit", type=int, default=100,
                        help="Hard iteration cap (default: 100)")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress per-iteration output")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("-n", "--name", type=str, default="diag",
                        help="System name (default: diag)")

    args = parser.parse_args()
    run_diagnostic(system_name=args.name, utilization=args.utilization,
                   seed=args.seed, sigma=args.sigma,
                   patience=args.patience, limit=args.limit,
                   verbose=not args.quiet)
