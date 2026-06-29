"""
Correlation study: Holistic FP vs. linearized surrogates (V1-V4).

Generates 100 random systems, runs Holistic FP and four linearized variants
on each, and measures their correlation. Produces scatter plots and CSV.
"""
import os
import csv
from random import Random

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from examples.example_models import get_system
from examples.generator import set_utilization
from model.analysis_function import calculate_priorities, init_wcrt, reset_wcrt
from model.linear_system import SchedulerType

from workspace.holistic_linearization.linearized_fp import (
    LinearizedFPAnalysis,
    LinearizedFPAnalysisV2,
    LinearizedFPAnalysisV3,
    LinearizedFPAnalysisV4,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def system_metrics(system):
    flows = system.flows
    wcrts = [f.wcrt for f in flows if f.wcrt is not None]
    deadlines = [f.deadline for f in flows]

    invslack = max((w - d) / d for w, d in zip(wcrts, deadlines))
    slacks = [(d - w) / d for w, d in zip(wcrts, deadlines)]
    system_slack = min(slacks)
    avg_wcrt = sum(wcrts) / len(wcrts) if wcrts else float('inf')
    max_wcrt = max(wcrts) if wcrts else float('inf')
    schedulable = all(w <= d for w, d in zip(wcrts, deadlines))

    return {
        'invslack': invslack,
        'system_slack': system_slack,
        'avg_wcrt': avg_wcrt,
        'max_wcrt': max_wcrt,
        'schedulable': schedulable,
    }


def run_analysis(system, analysis):
    reset_wcrt(system)
    init_wcrt(system)
    analysis.apply(system)
    return system_metrics(system)


def generate_systems(n_systems=100, size=(3, 4, 3), util_min=0.5, util_max=0.95):
    systems = []
    for i in range(n_systems):
        seed = 1000 + i
        r = Random(seed)
        util = util_min + (util_max - util_min) * i / max(n_systems - 1, 1)
        name = f"sys_{i}"
        s = get_system(size, r, name=name,
                       deadline_factor_min=0.5, deadline_factor_max=1,
                       sched=SchedulerType.FP, balanced=True)
        set_utilization(s, util)
        systems.append(s)
    return systems


ANALYSES = {
    'holistic':  HolisticFPAnalysis(limit_factor=10, reset=False),
    'V1_iter':   LinearizedFPAnalysis(limit_factor=10, alpha=0.0),
    'V2_2pass':  LinearizedFPAnalysisV2(limit_factor=10, n_passes=2),
    'V2_5pass':  LinearizedFPAnalysisV2(limit_factor=10, n_passes=5),
    'V3_oneshot': LinearizedFPAnalysisV3(limit_factor=10),
}


def run_study():
    print("Generating 100 random systems ...")
    systems = generate_systems(n_systems=100)
    results = []

    for i, sys in enumerate(systems):
        calculate_priorities(sys)

        row = {
            'id': i, 'seed': 1000 + i, 'name': sys.name,
            'n_flows': len(sys.flows), 'n_tasks': len(sys.tasks),
            'n_procs': len(sys.processors), 'utilization': sys.utilization,
        }

        for label, analysis in ANALYSES.items():
            m = run_analysis(sys, analysis)
            for metric in ['invslack', 'system_slack', 'avg_wcrt', 'max_wcrt', 'schedulable']:
                row[f'{label}_{metric}'] = m[metric]

        results.append(row)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(systems)} systems analyzed")

    csv_path = os.path.join(OUTPUT_DIR, "correlation_results.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")

    create_plots(results)
    print_summary(results)


def create_plots(results):
    variant_labels = [k for k in ANALYSES if k != 'holistic']
    metrics = [
        ('avg_wcrt', 'Average Flow WCRT'),
        ('system_slack', 'System Slack'),
        ('invslack', 'Inverse Slack'),
    ]

    fig, axes = plt.subplots(len(variant_labels), len(metrics),
                             figsize=(5 * len(metrics), 4.5 * len(variant_labels)),
                             squeeze=False)

    for vi, label in enumerate(variant_labels):
        for mi, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[vi][mi]
            x = np.array([r[f'holistic_{metric_key}'] for r in results])
            y = np.array([r[f'{label}_{metric_key}'] for r in results])

            mask = np.isfinite(x) & np.isfinite(y)
            x_f, y_f = x[mask], y[mask]

            if len(x_f) >= 3:
                r_pearson, _ = pearsonr(x_f, y_f)
                r_spearman, _ = spearmanr(x_f, y_f)
                slope, intercept = np.polyfit(x_f, y_f, 1)
            else:
                r_pearson = r_spearman = slope = intercept = float('nan')

            ax.scatter(x, y, alpha=0.6, s=18, color='#2166ac', edgecolors='none')

            if np.isfinite(slope):
                sx = np.sort(x_f)
                ax.plot(sx, slope * sx + intercept, color='#d6604d',
                        linewidth=1.5, linestyle='--')

            all_vals = np.concatenate([x_f, y_f])
            lo, hi = all_vals.min(), all_vals.max()
            margin = (hi - lo) * 0.1
            ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                    color='grey', linewidth=0.8, alpha=0.5)

            ax.set_xlabel('Holistic FP')
            ax.set_ylabel(label)
            ax.set_title(f'{metric_label}\nr_P={r_pearson:.4f}  r_S={r_spearman:.4f}',
                         fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, "correlation_scatter.png")
    fig.savefig(png_path, dpi=150)
    print(f"Scatter plots saved to {png_path}")


def print_summary(results):
    variant_labels = [k for k in ANALYSES if k != 'holistic']
    metrics = ['invslack', 'system_slack', 'avg_wcrt']

    print("\n" + "=" * 85)
    print("CORRELATION SUMMARY  (vs Holistic FP)")
    print("=" * 85)
    print(f"{'Variant':<16} {'Metric':<14} {'Pearson r':>10} {'Spearman rho':>12} {'p-value':>10}  {'NRMSE':>8}")
    print("-" * 85)

    for label in variant_labels:
        for metric in metrics:
            x = np.array([r[f'holistic_{metric}'] for r in results])
            y = np.array([r[f'{label}_{metric}'] for r in results])
            mask = np.isfinite(x) & np.isfinite(y)
            x_f, y_f = x[mask], y[mask]

            if len(x_f) >= 3:
                r_p, _ = pearsonr(x_f, y_f)
                r_s, p_s = spearmanr(x_f, y_f)
                nrmse = np.sqrt(np.mean((y_f - x_f) ** 2)) / (np.std(x_f) + 1e-10)
                print(f"{label:<16} {metric:<14} {r_p:>10.4f} {r_s:>12.4f} {p_s:>10.2e}  {nrmse:>8.4f}")
            else:
                print(f"{label:<16} {metric:<14} {'N/A':>10} {'N/A':>12}")

    print("-" * 85)
    sched = sum(1 for r in results if r['holistic_schedulable'])
    print(f"\nHolistic schedulable: {sched}/{len(results)}")

    for label in variant_labels:
        sched_v = sum(1 for r in results if r[f'{label}_schedulable'])
        print(f"{label} schedulable: {sched_v}/{len(results)}")


if __name__ == '__main__':
    run_study()
