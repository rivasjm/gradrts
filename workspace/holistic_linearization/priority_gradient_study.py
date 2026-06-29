"""
Priority-gradient validation study.

For each system:
1. Compute ∂(avg_wcrt)/∂s_i via V3 soft-priority (DM-based initial scores).
2. Use the gradient to pick the task that would benefit most from a
   priority promotion / demotion.
3. Apply the swap to the real Holistic FP and measure improvement.
4. Compare against random swaps as a baseline.
"""
import os, csv
from random import Random

import numpy as np
import torch

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from examples.example_models import get_system
from examples.generator import set_utilization
from model.analysis_function import calculate_priorities, init_wcrt, reset_wcrt
from model.linear_system import SchedulerType, Task
from workspace.holistic_linearization.differentiable_v3 import v3_soft_priority

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
TAU = 0.5
N_SWAPS = 1


def holistic_avg_wcrt(system) -> float:
    reset_wcrt(system)
    init_wcrt(system)
    HolisticFPAnalysis(limit_factor=10, reset=False).apply(system)
    wcrts = [f.wcrt for f in system.flows if f.wcrt is not None]
    return sum(wcrts) / len(wcrts) if wcrts else float('inf')


def v3_score_gradient(system) -> tuple[np.ndarray, float] | None:
    """Compute ∂(avg_wcrt)/∂s_i via V3 soft-priority.

    Returns (gradient, avg_wcrt_base) or None on failure.
    """
    tasks = system.tasks
    n = len(tasks)
    t2i = {t: i for i, t in enumerate(tasks)}

    pred_idx = [-1] * n
    for i, t in enumerate(tasks):
        p = t.predecessors
        pred_idx[i] = t2i[p[0]] if p else -1

    same_proc = torch.zeros(n, n, dtype=torch.bool)
    for i, ti in enumerate(tasks):
        for j, tj in enumerate(tasks):
            if ti.processor == tj.processor:
                same_proc[i, j] = True

    C = torch.tensor([t.wcet for t in tasks], dtype=torch.float64)
    T = torch.tensor([t.period for t in tasks], dtype=torch.float64)

    try:
        s_init = torch.tensor([t.priority for t in tasks], dtype=torch.float64)
        r, s_leaf = v3_soft_priority(C, T, pred_idx, same_proc, tau=TAU, s_init=s_init)
    except Exception:
        return None

    last_idx = [i for i, t in enumerate(tasks) if t.is_last]
    avg = r[last_idx].mean()
    avg.backward()
    g = s_leaf.grad

    if g is None:
        return None
    return g.detach().numpy(), float(avg.detach())


def apply_priority_swap(system, task_a, task_b):
    """Swap priorities of two tasks (must be on the same processor)."""
    pa, pb = task_a.priority, task_b.priority
    task_a.priority = pb
    task_b.priority = pa


def best_gradient_swap(system, gradient) -> tuple[Task, Task] | None:
    """Find best task to demote and best task to promote on each processor.

    Returns (demote_task, promote_task) — these two are swapped — or None.
    """
    tasks = system.tasks
    proc_tasks = {}
    for i, t in enumerate(tasks):
        proc_tasks.setdefault(t.processor.name, []).append((i, t))

    best_improvement = 0.0
    best_pair = None

    for pname, pts in proc_tasks.items():
        if len(pts) < 2:
            continue
        idxs = [i for i, _ in pts]
        ts = [t for _, t in pts]
        grads = np.array([gradient[i] for i in idxs])

        # Task with most negative gradient: benefits from LOWER priority (demote)
        # Task with most positive gradient: benefits from HIGHER priority (promote)
        demote_i = int(np.argmin(grads))
        promote_i = int(np.argmax(grads))

        if demote_i == promote_i:
            continue

        improvement = grads[promote_i] - grads[demote_i]
        if improvement > best_improvement:
            best_improvement = improvement
            best_pair = (ts[demote_i], ts[promote_i])

    return best_pair


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


def save_priorities(system):
    return [(t, t.priority) for t in system.tasks]


def restore_priorities(saved):
    for t, p in saved:
        t.priority = p


def run():
    print("Generating systems ...")
    systems = generate_systems(n=100)
    results = []

    for i, s in enumerate(systems):
        calculate_priorities(s)
        f0 = holistic_avg_wcrt(s)

        g_result = v3_score_gradient(s)
        if g_result is None:
            if (i + 1) % 20 == 0:
                print(f"  {i+1:>3}/{len(systems)}  (V3 failures so far: {sum(1 for r in results if r.get('v3_failed'))})")
            results.append({'id': i, 'name': s.name, 'util': s.utilization,
                           'f0': f0, 'v3_failed': True})
            continue

        gradient, v3_base = g_result

        # --- gradient-guided swap ---
        saved = save_priorities(s)
        pair = best_gradient_swap(s, gradient)

        if pair is None:
            restore_priorities(saved)
            results.append({'id': i, 'name': s.name, 'util': s.utilization,
                           'f0': f0, 'v3_failed': True})
            continue

        apply_priority_swap(s, pair[0], pair[1])
        f_guided = holistic_avg_wcrt(s)
        restore_priorities(saved)

        # --- random swap baseline (average of 5 random swaps) ---
        random_deltas = []
        n_proc = len(s.processors)
        for _ in range(5):
            saved2 = save_priorities(s)
            proc_tasks = {}
            for t in s.tasks:
                proc_tasks.setdefault(t.processor.name, []).append(t)
            # pick a random processor with >=2 tasks
            procs = [p for p, ts in proc_tasks.items() if len(ts) >= 2]
            if not procs:
                restore_priorities(saved2)
                random_deltas.append(0.0)
                continue
            pname = procs[Random(i * 1000 + _).randint(0, len(procs) - 1)]
            ts = proc_tasks[pname]
            rnd = Random(i * 2000 + _)
            a, b = rnd.sample(ts, 2)
            apply_priority_swap(s, a, b)
            fr = holistic_avg_wcrt(s)
            random_deltas.append(fr - f0)
            restore_priorities(saved2)

        delta_guided = f_guided - f0
        delta_random = np.mean(random_deltas)

        results.append({
            'id': i, 'name': s.name, 'util': s.utilization,
            'f0': f0, 'v3_failed': False,
            'v3_base': v3_base,
            'f_guided': f_guided,
            'delta_guided': delta_guided,
            'delta_random': delta_random,
            'improvement': delta_random - delta_guided,  # positive = better than random
            'demote': pair[0].name if pair else '',
            'promote': pair[1].name if pair else '',
        })

        if (i + 1) % 10 == 0:
            valid = [r for r in results if not r.get('v3_failed')]
            if valid:
                avg_imp = np.mean([r['improvement'] for r in valid])
                better = sum(1 for r in valid if r['delta_guided'] < r['delta_random'])
                print(f"  {i+1:>3}/{len(systems)}  better_than_random={better}/{len(valid)}  "
                      f"avg_improv={avg_imp:.4f}")

    # Save
    csv_path = os.path.join(OUTPUT_DIR, "priority_gradient_results.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV saved to {csv_path}")

    print_summary(results)


def print_summary(results):
    valid = [r for r in results if not r.get('v3_failed')]
    deltas_g = [r['delta_guided'] for r in valid]
    deltas_r = [r['delta_random'] for r in valid]
    improvements = [r['improvement'] for r in valid]
    better = sum(1 for r in valid if r['delta_guided'] < r['delta_random'])

    print("\n" + "=" * 60)
    print("PRIORITY GRADIENT VALIDATION")
    print("=" * 60)
    print(f"Valid systems:          {len(valid)} / {len(results)}")
    print(f"Guided better than random: {better} / {len(valid)} "
          f"({better/len(valid)*100:.1f}%)")
    print(f"Mean improvement over random: {np.mean(improvements):.4f}")
    print(f"Median improvement:           {np.median(improvements):.4f}")
    print(f"Mean Δ guided:  {np.mean(deltas_g):.4f}")
    print(f"Mean Δ random:  {np.mean(deltas_r):.4f}")
    print("=" * 60)

    # By utilization band
    utils = np.array([r['util'] for r in valid])
    imps = np.array(improvements)
    bands = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    print(f"\n{'Band':<14} {'n':>4} {'%better':>8}  {'mean_improv':>12}")
    print("-" * 45)
    for lo, hi in bands:
        mask = (utils >= lo) & (utils < hi)
        if mask.sum() > 0:
            b = sum(1 for r, m in zip(valid, mask) if m and r['delta_guided'] < r['delta_random'])
            print(f"{lo:.1f}-{hi:.1f}        {mask.sum():>4} {b/mask.sum():>7.1%}  "
                  f"{imps[mask].mean():>12.4f}")


if __name__ == '__main__':
    run()
