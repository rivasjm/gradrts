"""
Gradient correlation study: V3 autograd vs Holistic FP finite differences.

For each system, compares the gradient of avg_wcrt w.r.t. WCET:
- Holistic FP: central finite differences
- V3: PyTorch autograd

Reports cosine similarity between the two gradient vectors.
"""
import os
import csv
from random import Random

import numpy as np
import matplotlib.pyplot as plt

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from examples.example_models import get_system
from examples.generator import set_utilization
from model.analysis_function import calculate_priorities, init_wcrt, reset_wcrt
from model.linear_system import SchedulerType

from workspace.holistic_linearization.differentiable_v3 import v3_forward_torch

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-5


def holistic_fd_gradient(system) -> np.ndarray | None:
    """Central finite-difference gradient of avg_wcrt w.r.t. each WCET."""
    tasks = system.tasks
    n = len(tasks)
    analyser = HolisticFPAnalysis(limit_factor=10, reset=False)

    def _avg_wcrt():
        reset_wcrt(system)
        init_wcrt(system)
        analyser.apply(system)
        wcrts = [f.wcrt for f in system.flows if f.wcrt is not None]
        return sum(wcrts) / len(wcrts) if wcrts else float('inf')

    f0 = _avg_wcrt()
    if not np.isfinite(f0):
        return None

    orig_wcets = [t.wcet for t in tasks]
    grad = np.zeros(n)

    for i in range(n):
        tasks[i].wcet = orig_wcets[i] + EPS
        fp = _avg_wcrt()

        tasks[i].wcet = orig_wcets[i] - EPS
        fm = _avg_wcrt()

        tasks[i].wcet = orig_wcets[i]

        if np.isfinite(fp) and np.isfinite(fm):
            grad[i] = (fp - fm) / (2 * EPS)
        else:
            grad[i] = np.nan

    return grad


def v3_autograd_gradient(system) -> np.ndarray | None:
    """PyTorch autograd gradient of V3 avg_wcrt w.r.t. WCET."""
    tasks = system.tasks
    last_idx = []
    for flow in system.flows:
        for t in flow.tasks:
            if t.is_last:
                for j, tt in enumerate(tasks):
                    if tt is t:
                        last_idx.append(j)
                        break
                break

    try:
        r, C_leaf = v3_forward_torch(system)
    except Exception:
        return None

    avg = r[last_idx].mean()
    avg.backward()
    g = C_leaf.grad

    if g is None:
        return None
    return g.detach().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def generate_systems(n=100, size=(3, 4, 3), util_min=0.5, util_max=0.95):
    systems = []
    for i in range(n):
        r = Random(1000 + i)
        util = util_min + (util_max - util_min) * i / max(n - 1, 1)
        s = get_system(size, r, name=f'sys_{i}',
                       deadline_factor_min=0.5, deadline_factor_max=1,
                       sched=SchedulerType.FP, balanced=True)
        set_utilization(s, util)
        systems.append(s)
    return systems


def run():
    print("Generating systems ...")
    systems = generate_systems(n=100)
    results = []

    for i, s in enumerate(systems):
        calculate_priorities(s)

        g_hol = holistic_fd_gradient(s)
        g_v3 = v3_autograd_gradient(s)

        if g_hol is not None and g_v3 is not None and len(g_hol) == len(g_v3):
            mask = np.isfinite(g_hol) & np.isfinite(g_v3)
            cos = cosine_similarity(g_hol[mask], g_v3[mask])
            results.append({
                'id': i, 'name': s.name, 'util': s.utilization,
                'cos_sim': cos,
                'norm_hol': np.linalg.norm(g_hol[mask]),
                'norm_v3': np.linalg.norm(g_v3[mask]),
                'n_finite': mask.sum(),
            })
        else:
            results.append({
                'id': i, 'name': s.name, 'util': s.utilization,
                'cos_sim': np.nan, 'norm_hol': np.nan,
                'norm_v3': np.nan, 'n_finite': 0,
            })

        if (i + 1) % 10 == 0:
            valid = [r for r in results if not np.isnan(r['cos_sim'])]
            if valid:
                avg_cos = np.mean([r['cos_sim'] for r in valid])
                print(f"  {i+1:>3}/{len(systems)}  avg cos_sim = {avg_cos:.4f}  "
                      f"({len(valid)} valid)")

    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "gradient_correlation.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['id', 'name', 'util', 'cos_sim',
                                          'norm_hol', 'norm_v3', 'n_finite'])
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV saved to {csv_path}")

    # Plot
    plot_results(results)
    print_summary(results)


def plot_results(results):
    valid = [r for r in results if not np.isnan(r['cos_sim'])]
    if not valid:
        print("No valid results to plot")
        return

    utils = [r['util'] for r in valid]
    cos = [r['cos_sim'] for r in valid]
    norms_hol = [r['norm_hol'] for r in valid]
    norms_v3 = [r['norm_v3'] for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Histogram of cosine similarities
    axes[0].hist(cos, bins=20, color='#2166ac', edgecolor='white', alpha=0.8)
    axes[0].axvline(np.mean(cos), color='#d6604d', linestyle='--', linewidth=2,
                    label=f'mean = {np.mean(cos):.4f}')
    axes[0].set_xlabel('Cosine similarity')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Gradient direction alignment\nV3 autograd vs Holistic FD')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Norm comparison (magnitude)
    axes[1].scatter(norms_hol, norms_v3, alpha=0.6, s=15, color='#2166ac')
    axes[1].set_xlabel('||grad Holistic||')
    axes[1].set_ylabel('||grad V3||')
    axes[1].set_title('Gradient magnitude')
    axes[1].grid(True, alpha=0.3)
    lo = min(min(norms_hol), min(norms_v3))
    hi = max(max(norms_hol), max(norms_v3))
    axes[1].plot([lo, hi], [lo, hi], 'grey', linewidth=0.8, alpha=0.5)

    # Cosine vs utilization
    axes[2].scatter(utils, cos, alpha=0.6, s=15, color='#d6604d')
    axes[2].set_xlabel('System utilization')
    axes[2].set_ylabel('cosine similarity')
    axes[2].set_title('Alignment vs utilization')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-1.05, 1.05)

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "gradient_correlation.png")
    fig.savefig(png_path, dpi=150)
    print(f"Plot saved to {png_path}")


def print_summary(results):
    valid = [r for r in results if not np.isnan(r['cos_sim'])]
    cos_vals = np.array([r['cos_sim'] for r in valid])

    print("\n" + "=" * 55)
    print("GRADIENT CORRELATION SUMMARY")
    print("=" * 55)
    print(f"Valid systems:           {len(valid)} / {len(results)}")
    print(f"Mean cosine similarity:  {np.mean(cos_vals):.4f}")
    print(f"Median cosine similarity:{np.median(cos_vals):.4f}")
    print(f"Std cosine similarity:   {np.std(cos_vals):.4f}")
    print(f"Min cosine similarity:   {np.min(cos_vals):.4f}")
    print(f"Max cosine similarity:   {np.max(cos_vals):.4f}")
    print(f"Fraction > 0.8:          {(cos_vals > 0.8).mean():.2%}")
    print(f"Fraction > 0.9:          {(cos_vals > 0.9).mean():.2%}")
    print(f"Fraction > 0.95:         {(cos_vals > 0.95).mean():.2%}")
    print("=" * 55)


if __name__ == '__main__':
    run()
