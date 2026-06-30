"""Direct comparison: Sequential (FD) vs Surrogate gradient descent on EDF systems.

Measures:
  - Final cost after optimisation
  - Schedulability rate
  - Runtime per system
  - Cost trajectory
"""

import copy
import time
import sys
from random import Random
import numpy as np

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import SequentialGradientFunction
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import DeadlineExtractor
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem, SchedulerType
from model.linear_system_utils import backup_assignment, restore_assignment
from surrogate.surrogate_edf import SurrogateEDFGradient


def optimise(
    system: LinearSystem,
    gradient_function,
    max_iters: int = 50,
    patience: int = 15,
    verbose: bool = False,
):
    """Run gradient-descent optimisation on *system* (mutated in-place).
    Returns (final_cost, trajectory, elapsed_seconds).
    """
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
    stop_fn = ThresholdStopFunction(limit=max_iters, patience=patience)
    update_fn = NoisyAdam()

    optimizer = GradientDescentOptimizer(
        parameter_handler=ph,
        cost_function=cost_fn,
        stop_function=stop_fn,
        gradient_function=gradient_function,
        update_function=update_fn,
        verbose=verbose,
    )

    PDAssignment().apply(system)

    # Record cost trajectory (cheap: only runs the real analysis)
    x = ph.extract(system)
    trajectory = []

    def record_cb(t_, S_, x_, xb_, cost_, best_, ref_cost_):
        trajectory.append(cost_)

    optimizer.callback = record_cb  # set before apply

    t0 = time.perf_counter()
    optimizer.apply(system)
    elapsed = time.perf_counter() - t0

    # Final schedulability check (use limit_factor=10 to get actual WCRT values)
    final_analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    final_analysis.apply(system)
    schedulable = system.is_schedulable()
    # Recompute cost from final state (reliable)
    if schedulable:
        final_cost = max((f.wcrt - f.deadline) / max(f.deadline, 1e-9)
                         for f in system.flows)
    else:
        final_cost = trajectory[-1] if trajectory else float("inf")

    return final_cost, trajectory, elapsed, schedulable


def compare_on_systems(systems, max_iters=50, patience=15):
    """Run both methods on identical copies of each system."""
    results_seq = []
    results_surr = []

    for i, sys_template in enumerate(systems):
        # Sequential (FD) gradient
        sys_seq = copy.deepcopy(sys_template)
        grad_seq = SequentialGradientFunction(
            cost_function=InvslackCost(
                parameter_handler=DeadlineExtractor(),
                analysis=HolisticLocalEDFAnalysis(limit_factor=10, reset=False),
            ),
            sigma=1.5,
        )
        fc_seq, traj_seq, t_seq, sched_seq = optimise(
            sys_seq, grad_seq, max_iters=max_iters, patience=patience
        )
        results_seq.append((fc_seq, traj_seq, t_seq, sched_seq))

        # Surrogate gradient
        sys_surr = copy.deepcopy(sys_template)
        grad_surr = SurrogateEDFGradient(
            tau=0.5, N_w=10, N_jitter=2, M_psi=50,
            temperature_max=0.1, grad_clip=1.0,
        )
        fc_surr, traj_surr, t_surr, sched_surr = optimise(
            sys_surr, grad_surr, max_iters=max_iters, patience=patience
        )
        results_surr.append((fc_surr, traj_surr, t_surr, sched_surr))

        print(f"  sys {i:2d}: "
              f"Seq cost={fc_seq:+.3f} sched={sched_seq} t={t_seq:.1f}s | "
              f"Surr cost={fc_surr:+.3f} sched={sched_surr} t={t_surr:.1f}s | "
              f"Δcost={fc_surr - fc_seq:+.3f}  speedup={t_seq / max(t_surr, 0.001):.1f}x",
              flush=True)

    return results_seq, results_surr


def main():
    rnd = Random(42)
    size = (3, 4, 3)
    n_systems = 8  # keep small for runtime

    print("=" * 70)
    print("GDPA-Seq (FD)  vs  GDPA-Surr (differentiable surrogate)")
    print(f"  Systems: {n_systems}, size: {size}, max_iters=50, patience=15")
    print("=" * 70)

    systems = [
        get_system(size, rnd, balanced=True, name=str(i),
                   deadline_factor_min=0.3, sched=SchedulerType.EDF,
                   deadline_factor_max=0.7)
        for i in range(n_systems)
    ]

    # Compute PD baseline cost for each
    print("\nPD baseline:")
    pd_costs = []
    pd_sched = []
    for i, sys_ in enumerate(systems):
        sys_copy = copy.deepcopy(sys_)
        PDAssignment().apply(sys_copy)
        ana = HolisticLocalEDFAnalysis(limit_factor=1, reset=True)
        ana.apply(sys_copy)
        sched = sys_copy.is_schedulable()
        if sched:
            cost = max((f.wcrt - f.deadline) / max(f.deadline, 1e-9)
                       for f in sys_copy.flows)
        else:
            cost = float("inf")
        pd_costs.append(cost)
        pd_sched.append(sched)
        print(f"  sys {i}: PD cost={cost:+.3f}  sched={sched}")
    print(f"  PD schedulable: {sum(pd_sched)}/{n_systems}")

    print("\nOptimisation comparison:")
    res_seq, res_surr = compare_on_systems(systems, max_iters=50, patience=15)

    # --- Summary ---
    fc_seq = np.array([r[0] for r in res_seq])
    fc_surr = np.array([r[0] for r in res_surr])
    t_seq = np.array([r[2] for r in res_seq])
    t_surr = np.array([r[2] for r in res_surr])
    sched_seq = sum(r[3] for r in res_seq)
    sched_surr = sum(r[3] for r in res_surr)

    # Improvement over PD
    pd_costs_arr = np.array(pd_costs)
    imp_seq = pd_costs_arr - fc_seq   # positive = improved
    imp_surr = pd_costs_arr - fc_surr

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  PD schedulable:            {sum(pd_sched)}/{n_systems}")
    print(f"  GDPA-Seq schedulable:      {sched_seq}/{n_systems}")
    print(f"  GDPA-Surr schedulable:     {sched_surr}/{n_systems}")
    print()
    print(f"  GDPA-Seq  final cost:      {np.mean(fc_seq):+.4f}  (±{np.std(fc_seq):.4f})")
    print(f"  GDPA-Surr final cost:      {np.mean(fc_surr):+.4f}  (±{np.std(fc_surr):.4f})")
    print(f"  GDPA-Seq  improvement:     {np.mean(imp_seq):+.4f}")
    print(f"  GDPA-Surr improvement:     {np.mean(imp_surr):+.4f}")
    print()
    print(f"  GDPA-Seq  time/system:     {np.mean(t_seq):.2f}s  (±{np.std(t_seq):.2f})")
    print(f"  GDPA-Surr time/system:     {np.mean(t_surr):.2f}s  (±{np.std(t_surr):.2f})")
    print(f"  Speed-up:                  {np.mean(t_seq) / max(np.mean(t_surr), 0.001):.1f}x")

    # Per-system breakdown
    print("\n  Per-system Δ (Surr − Seq):")
    for i, (s, u) in enumerate(zip(fc_seq, fc_surr)):
        better = "SURR" if u < s else ("SEQ" if s < u else "tie")
        print(f"    sys {i}: Δ={u - s:+.4f}  ({better})  "
              f"t_seq={t_seq[i]:.1f}s  t_surr={t_surr[i]:.1f}s")

    print("\nDone.")


if __name__ == "__main__":
    main()
