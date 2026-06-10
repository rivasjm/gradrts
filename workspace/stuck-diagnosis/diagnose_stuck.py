"""
Diagnose why the mapping-only gradient evaluation appears stuck.

Replicates exactly the setup of gradient_fp_mapping_only_val.py on a single
system but instruments every step with timing and progress prints so the
bottleneck is immediately visible.
"""
import numpy as np
import sys
import time
from datetime import datetime
from random import Random

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from examples.generator import set_utilization, unbalance
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import MappingOnlyExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem
from model.linear_system_utils import backup_assignment, restore_assignment
from vector.vector_fp import VectorFPGradientFunction, MappingOnlyMatrix


def timed(msg, thunk):
    """Run *thunk*, print *msg* and elapsed time, return result."""
    sys.stdout.write(f"  {msg} ... ")
    sys.stdout.flush()
    t0 = time.perf_counter()
    result = thunk()
    elapsed = time.perf_counter() - t0
    print(f"{elapsed:.3f}s")
    return result


def progress_callback(t, S, x, xb, cost, best, ref_cost):
    """Print a one-line summary each iteration."""
    imp = "*" if cost < best else " "
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] iter {t:3d}{imp}  "
          f"cost={cost:.4f}  best={best:.4f}")


def run_single_system(seed=42, utilization=0.5, verbose=True):
    """Run the exact same optimisation as gdpa_pd_fp_mapping_only_vector,
    with timing on every phase."""

    print("=" * 60)
    print(f"Setup: seed={seed}, u={utilization}")
    print("=" * 60)

    # --- build system (same as eval script) ---
    rnd = Random(seed)
    size = (3, 4, 3)
    system = timed("Generating system",
                   lambda: get_system(size, rnd, balanced=False, name=str(seed),
                                      deadline_factor_min=0.5, deadline_factor_max=1))
    print(f"  {len(system.tasks)} tasks, {len(system.processors)} procs, "
          f"{len(system.flows)} flows")

    timed("Applying unbalance", lambda: unbalance(system))
    timed("Setting utilisation", lambda: set_utilization(system, utilization))

    # --- PD assignment ---
    timed("PD assignment", lambda: PDAssignment(normalize=True).apply(system))

    # --- build solver (exactly matches evaluation) ---
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    parameter_handler = MappingOnlyExtractor()
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    stop_function = ThresholdStopFunction(limit=100, patience=20)
    gradient_function = VectorFPGradientFunction(
        scenarios_builder=MappingOnlyMatrix(), sigma=3.0, cost_limit_factor=3)
    update_function = NoisyAdam(lr=1.5, beta1=0.9, beta2=0.999, epsilon=0.1, gamma=0.5)

    optimizer = GradientDescentOptimizer(
        parameter_handler=parameter_handler,
        cost_function=cost_function,
        stop_function=stop_function,
        gradient_function=gradient_function,
        update_function=update_function,
        callback=progress_callback,
        verbose=verbose)

    # --- run optimisation ---
    print()
    print("Starting gradient descent (up to 100 iters, patience=20, sigma=3.0)")
    print("-" * 60)
    t_total = time.perf_counter()

    try:
        solution = optimizer.apply(system)
    except KeyboardInterrupt:
        print("\nINTERRUPTED by user")
        return
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    elapsed = time.perf_counter() - t_total
    print("-" * 60)
    print(f"Optimisation finished in {elapsed:.2f}s")

    # --- final analysis ---
    timed("Final analysis (limit_factor=1)",
          lambda: HolisticFPAnalysis(limit_factor=1, reset=True).apply(system))
    schedulable = system.is_schedulable()
    print(f"Result: {'SCHEDULABLE' if schedulable else 'UNSCHEDULABLE'}")
    print(f"Final cost: {stop_function.solution_cost():.4f}")


def profile_first_iteration(seed=42, utilization=0.5):
    """Drill into the first gradient-descent iteration to isolate the slow step."""

    print("=" * 60)
    print(f"First-iteration profile: seed={seed}, u={utilization}")
    print("=" * 60)

    rnd = Random(seed)
    size = (3, 4, 3)
    system = get_system(size, rnd, balanced=False, name=str(seed),
                        deadline_factor_min=0.5, deadline_factor_max=1)
    unbalance(system)
    set_utilization(system, utilization)
    PDAssignment(normalize=True).apply(system)

    parameter_handler = MappingOnlyExtractor()
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    cost_function = InvslackCost(parameter_handler=parameter_handler, analysis=analysis)
    gradient_function = VectorFPGradientFunction(
        scenarios_builder=MappingOnlyMatrix(), sigma=3.0, cost_limit_factor=3)

    x = parameter_handler.extract(system)
    print(f"Parameter count: {len(x)} (= {len(system.processors)} procs × "
          f"{len(system.tasks)} tasks)")

    # --- cost computation ---
    print()
    a = backup_assignment(system)
    timed("  cost: parameter_handler.insert",
          lambda: parameter_handler.insert(system, x))
    timed("  cost: HolisticFPAnalysis.apply (limit=10)",
          lambda: analysis.apply(system))

    cost = max([(flow.wcrt - flow.deadline) / flow.deadline for flow in system.flows])
    print(f"  Cost = {cost:.4f}")
    restore_assignment(system, a)

    # --- gradient computation ---
    print()
    t0 = time.perf_counter()
    deltas = gradient_function.delta_function.apply(system, x)
    print(f"  grad: sigma={gradient_function.delta_function.sigma}, "
          f"delta={deltas[0]:.6f}, {len(x)} params → {2*len(x)} scenarios")

    inputs = timed("  grad: gradient_inputs_from_deltas",
                   lambda: _make_inputs(x, deltas))

    timed("  grad: scenarios_builder.apply (build priority matrices)",
          lambda: gradient_function.scenarios_builder.apply(system, inputs))

    print("  grad: VectorHolisticFPAnalysis.apply (cost_limit_factor=3) ... ", end="")
    sys.stdout.flush()
    t1 = time.perf_counter()
    costs = gradient_function._compute_costs(system, inputs)
    t2 = time.perf_counter()
    print(f"{t2-t1:.3f}s")

    t_total = time.perf_counter() - t0
    print(f"  Total gradient computation: {t_total:.3f}s")


def _make_inputs(x, deltas):
    """Inline gradient_inputs_from_deltas to avoid import issues."""
    ret = []
    for i in range(len(x)):
        v = x[:]
        v[i] += deltas[i]
        ret.append(v)
        v = x[:]
        v[i] -= deltas[i]
        ret.append(v)
    return ret


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Diagnose stuck/slow mapping-only gradient evaluation")
    parser.add_argument("--profile", action="store_true",
                        help="Profile just the first iteration in detail")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-u", "--utilization", type=float, default=0.5)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    if args.profile:
        profile_first_iteration(seed=args.seed, utilization=args.utilization)
    else:
        run_single_system(seed=args.seed, utilization=args.utilization,
                          verbose=not args.quiet)
