"""Validation script: correlate surrogate vs real EDF analysis, and
surrogate gradient vs finite-difference gradient."""

import math
import sys
from random import Random

import numpy as np
import torch
import matplotlib.pyplot as plt

from analysis.holistic_local_edf_analysis import HolisticLocalEDFAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.gradient_function import (
    AvgSeparationDelta,
    SequentialGradientFunction,
    gradient_inputs_from_deltas,
)
from gradient_descent.parameter_handlers import DeadlineExtractor
from model.linear_system import LinearSystem, SchedulerType
from model.linear_system_utils import backup_assignment, restore_assignment
from surrogate.surrogate_edf import (
    surrogate_edf_forward,
    _build_system_tensors,
    SurrogateEDFGradient,
)


# =========================================================================
# 1. WCRT correlation: surrogate vs real analysis
# =========================================================================

def compare_wcrts(system: LinearSystem, tau=0.5, N_w=10):
    """Compare surrogate WCRTs against exact HolisticLocalEDFAnalysis."""
    real_analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False, verbose=False)
    real_analysis.apply(system)
    real_wcrts = np.array([t.wcrt for t in system.tasks])

    *_, max_d = _build_system_tensors(system)
    ph = DeadlineExtractor()
    x = ph.extract(system)
    s = torch.tensor(x, dtype=torch.float64, requires_grad=True)

    r_surr, _ = surrogate_edf_forward(
        system, s, tau=tau, N_w=N_w, N_jitter=2, M_psi=50, temperature_max=0.1
    )
    surr_wcrts = r_surr.detach().numpy()

    # Pearson correlation
    mask = np.isfinite(real_wcrts) & np.isfinite(surr_wcrts) & (real_wcrts > 0)
    if mask.sum() < 3:
        return None
    corr = np.corrcoef(real_wcrts[mask], surr_wcrts[mask])[0, 1]

    # Relative error
    rel_err = np.abs(surr_wcrts[mask] - real_wcrts[mask]) / real_wcrts[mask]
    mean_rel_err = np.mean(rel_err)

    return corr, mean_rel_err, real_wcrts, surr_wcrts


# =========================================================================
# 2. Gradient correlation: surrogate grad vs finite-difference grad
# =========================================================================

def compare_gradients(system: LinearSystem, tau=0.5, N_w=10, fd_sigma=1.5):
    """Compare surrogate gradient direction with finite-difference gradient."""
    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False, verbose=False)
    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)

    x = ph.extract(system)

    # --- finite-difference gradient ---
    fd_grad_fn = SequentialGradientFunction(cost_function=cost_fn, sigma=fd_sigma)
    fd_grad = np.array(fd_grad_fn.compute(system, x))

    # --- surrogate gradient ---
    surr_grad_fn = SurrogateEDFGradient(
        tau=tau, N_w=N_w, N_jitter=2, M_psi=50
    )
    surr_grad = np.array(surr_grad_fn.compute(system, x))

    # Cosine similarity
    fd_norm = np.linalg.norm(fd_grad)
    surr_norm = np.linalg.norm(surr_grad)
    if fd_norm < 1e-9 or surr_norm < 1e-9:
        return None
    cos_sim = np.dot(fd_grad, surr_grad) / (fd_norm * surr_norm)

    return cos_sim, fd_grad, surr_grad


# =========================================================================
# 3. Optimization test: can the surrogate-guided optimizer find schedulable
#    configurations better than PD?
# =========================================================================

def test_optimization(system: LinearSystem, tau=0.5, N_w=10, max_iters=50):
    """Run a few gradient descent steps and check if cost decreases."""
    from gradient_descent.gradient_optimizer import GradientDescentOptimizer
    from gradient_descent.stop_functions import FixedIterationsStop
    from gradient_descent.update_functions import NoisyAdam
    from model.linear_system_utils import backup_assignment, restore_assignment

    analysis = HolisticLocalEDFAnalysis(limit_factor=10, reset=False)
    ph = DeadlineExtractor()
    cost_fn = InvslackCost(parameter_handler=ph, analysis=analysis)
    stop_fn = FixedIterationsStop(iterations=max_iters)
    grad_fn = SurrogateEDFGradient(tau=tau, N_w=N_w, N_jitter=2, M_psi=50)
    update_fn = NoisyAdam()

    optimizer = GradientDescentOptimizer(
        parameter_handler=ph,
        cost_function=cost_fn,
        stop_function=stop_fn,
        gradient_function=grad_fn,
        update_function=update_fn,
        verbose=False,
    )

    # Initial assignment: PD
    PDAssignment().apply(system)
    init_x = ph.extract(system)
    analysis.apply(system)
    init_cost = max((f.wcrt - f.deadline) / max(f.deadline, 1e-9)
                    for f in system.flows)

    # Run optimizer
    optimizer.apply(system)
    final_cost = stop_fn.solution_cost()

    return init_cost, final_cost


# =========================================================================
# Main
# =========================================================================

def main():
    print("=" * 60)
    print("EDF Surrogate Validation")
    print("=" * 60)

    rnd = Random(42)
    size = (3, 4, 3)  # (flows, tasks/flow, processors)
    n_systems = 30

    print(f"Generating {n_systems} EDF systems  (size {size}) ...")
    systems = [
        get_system(size, rnd, balanced=True, name=str(i),
                   deadline_factor_min=0.3, sched=SchedulerType.EDF,
                   deadline_factor_max=0.7)
        for i in range(n_systems)
    ]

    # --- 1. WCRT correlation ---
    print("\n--- WCRT correlation (surrogate vs real) ---")
    wcrt_corrs = []
    wcrt_errors = []
    all_real = []
    all_surr = []
    for i, sys_ in enumerate(systems):
        PDAssignment().apply(sys_)
        res = compare_wcrts(sys_, tau=0.5, N_w=10)
        if res is not None:
            corr, rel_err, real_w, surr_w = res
            wcrt_corrs.append(corr)
            wcrt_errors.append(rel_err)
            all_real.extend(real_w.tolist())
            all_surr.extend(surr_w.tolist())

    if wcrt_corrs:
        print(f"  Mean Pearson r  = {np.mean(wcrt_corrs):.4f}  (±{np.std(wcrt_corrs):.4f})")
        print(f"  Mean rel. error = {np.mean(wcrt_errors):.4f}  (±{np.std(wcrt_errors):.4f})")

        # Scatter plot
        plt.figure(figsize=(5, 5))
        plt.scatter(all_real, all_surr, alpha=0.3, s=8)
        lims = [0, max(max(all_real), max(all_surr)) * 1.1]
        plt.plot(lims, lims, 'k--', alpha=0.5)
        plt.xlabel("Real WCRT")
        plt.ylabel("Surrogate WCRT")
        plt.title(f"WCRT correlation  (r={np.mean(wcrt_corrs):.3f})")
        plt.tight_layout()
        plt.savefig("workspace/surrogate_edf_local/wcrt_correlation.png", dpi=120)
        plt.close()
        print("  → saved wcrt_correlation.png")
    else:
        print("  No valid results (all systems unschedulable or failed).")

    # --- 2. Gradient correlation ---
    print("\n--- Gradient direction correlation (surrogate vs FD) ---")
    grad_cos = []
    for i, sys_ in enumerate(systems):
        PDAssignment().apply(sys_)
        # Run HolisticLocalEDFAnalysis first to set WCRTs
        ana = HolisticLocalEDFAnalysis(limit_factor=10, reset=False, verbose=False)
        ana.apply(sys_)
        res = compare_gradients(sys_, tau=0.5, N_w=10)
        if res is not None:
            cos_sim, fd_g, surr_g = res
            grad_cos.append(cos_sim)

    if grad_cos:
        print(f"  Mean cosine similarity = {np.mean(grad_cos):.4f}  (±{np.std(grad_cos):.4f})")
        print(f"  Fraction cos > 0       = {np.mean(np.array(grad_cos) > 0):.2%}")

        plt.figure(figsize=(5, 4))
        plt.hist(grad_cos, bins=20, edgecolor='k', alpha=0.7)
        plt.axvline(0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Cosine similarity (surrogate vs FD gradient)")
        plt.ylabel("Count")
        plt.title(f"Gradient alignment  (mean cos={np.mean(grad_cos):.3f})")
        plt.tight_layout()
        plt.savefig("workspace/surrogate_edf_local/gradient_correlation.png", dpi=120)
        plt.close()
        print("  → saved gradient_correlation.png")
    else:
        print("  No valid gradient results.")

    # --- 3. Optimisation smoke test ---
    print("\n--- Optimisation smoke test (5 systems) ---")
    for i, sys_ in enumerate(systems[:5]):
        PDAssignment().apply(sys_)
        res = test_optimization(sys_, tau=0.5, N_w=10, max_iters=30)
        init_c, final_c = res
        arrow = "↓" if final_c < init_c else "↑"
        print(f"  sys {i}: init_cost={init_c:.4f} → final_cost={final_c:.4f}  {arrow}")

    print("\nDone.")


if __name__ == "__main__":
    main()
