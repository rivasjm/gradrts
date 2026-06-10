"""
Diagnose brute-force mapping search progress.

Replicates bf_mapping_fp from gradient_fp_mapping_only_val.py on a single
system, printing per-batch progress so you can see whether it is making
headway or stuck in a single batch.
"""
import itertools
import math
import sys
import time
from datetime import datetime, timedelta

import numpy as np

from assignment.assignments import PDAssignment
from assignment.bf_assignment import BruteForceMappingAssignment
from examples.example_models import get_system
from examples.generator import set_utilization, unbalance
from random import Random


def run_bf_diagnostic(seed=42, utilization=0.5, batch_size=50):
    print("=" * 60)
    print(f"Brute-force diagnostic: seed={seed}, u={utilization}")
    print("=" * 60)

    # --- build system (exactly as the eval) ---
    rnd = Random(seed)
    size = (3, 4, 3)
    system = get_system(size, rnd, balanced=False, name=str(seed),
                        deadline_factor_min=0.5, deadline_factor_max=1)
    print(f"System: {len(system.tasks)} tasks, {len(system.processors)} procs, "
          f"{len(system.flows)} flows")

    unbalance(system)
    set_utilization(system, utilization)
    PDAssignment(normalize=True).apply(system)

    tasks = system.tasks
    n = len(tasks)
    procs = system.processors
    pi = len(procs)

    mapping_space_size = pi ** n
    print(f"Mapping space: {pi}^{n} = {mapping_space_size:,}")
    print(f"Batch size: {batch_size}  →  {math.ceil(mapping_space_size / batch_size):,} batches")
    print()

    import model.linear_system_utils as lsu
    from vector.vector_fp import VectorHolisticFPAnalysis

    analysis = VectorHolisticFPAnalysis(limit_factor=10)
    deadlines = np.array([t.flow.deadline for t in tasks]).reshape(n, 1)
    mapping_space = list(itertools.product(range(pi), repeat=n))

    pm_batch = []
    mappings_batch = []
    priorities_batch = []
    processed = 0
    schedulable = False
    batch_idx = 0
    t_start = time.perf_counter()
    t_batch_start = t_start

    for mapping_tuple in mapping_space:
        proc_tasks = [[] for _ in range(pi)]
        for task_idx, proc_idx in enumerate(mapping_tuple):
            proc_tasks[proc_idx].append(task_idx)

        candidate_priorities = [0.0] * n
        for proc_idx in range(pi):
            sorted_indices = sorted(
                proc_tasks[proc_idx],
                key=lambda idx: (tasks[idx].deadline if tasks[idx].deadline is not None else math.inf, idx)
            )
            k = len(sorted_indices)
            for prio_pos, idx in enumerate(sorted_indices, start=1):
                candidate_priorities[idx] = float(k - prio_pos + 1)

        max_prio = max(candidate_priorities) if candidate_priorities else 1.0
        if max_prio > 0:
            candidate_priorities = [p / max_prio for p in candidate_priorities]

        pm = _single_priority_matrix(mapping_tuple, candidate_priorities, n)
        pm_batch.append(pm)
        mappings_batch.append(mapping_tuple)
        priorities_batch.append(candidate_priorities)

        if len(pm_batch) == batch_size:
            processed += len(pm_batch)
            batch_idx += 1
            a = lsu.backup_assignment(system)
            t_analysis = time.perf_counter()
            pm_stack = np.stack(pm_batch, axis=0)
            analysis.apply(system, scenarios=pm_stack)
            r = analysis.scenarios_response_times
            slacks = deadlines - r
            batch_schedulable = np.all(slacks >= 0, axis=0)

            t_analysis_elapsed = time.perf_counter() - t_analysis
            if np.any(batch_schedulable):
                schedulable = True
                index = np.argmax(batch_schedulable)
                for task, proc_idx, prio in zip(tasks, mappings_batch[index], priorities_batch[index]):
                    task.processor = procs[proc_idx]
                    task.priority = prio
                lsu.restore_assignment(system, a)
                break
            lsu.restore_assignment(system, a)

            t_batch_elapsed = time.perf_counter() - t_batch_start
            t_total = time.perf_counter() - t_start
            pct = processed / mapping_space_size * 100
            now = datetime.now()

            # Estimate remaining time
            if batch_idx > 0:
                avg_per_batch = t_total / batch_idx
                remaining_batches = (mapping_space_size - processed) / batch_size
                eta = timedelta(seconds=avg_per_batch * remaining_batches)
            else:
                eta = "?"

            print(f"  [{now.strftime('%H:%M:%S')}] batch {batch_idx:5d}  "
                  f"processed {processed:7,}/{mapping_space_size:,} ({pct:5.1f}%)  "
                  f"batch {t_batch_elapsed:6.3f}s  (analysis {t_analysis_elapsed:5.3f}s)  "
                  f"total {t_total:.1f}s  ~ETA {eta}")

            pm_batch.clear()
            mappings_batch.clear()
            priorities_batch.clear()
            t_batch_start = time.perf_counter()

    # Process remaining partial batch
    if len(pm_batch) > 0 and not schedulable:
        processed += len(pm_batch)
        a = lsu.backup_assignment(system)
        pm_stack = np.stack(pm_batch, axis=0)
        analysis.apply(system, scenarios=pm_stack)
        r = analysis.scenarios_response_times
        slacks = deadlines - r
        batch_schedulable = np.all(slacks >= 0, axis=0)
        if np.any(batch_schedulable):
            schedulable = True
            index = np.argmax(batch_schedulable)
            for task, proc_idx, prio in zip(tasks, mappings_batch[index], priorities_batch[index]):
                task.processor = procs[proc_idx]
                task.priority = prio
        lsu.restore_assignment(system, a)

    elapsed = time.perf_counter() - t_start
    print()
    print("-" * 60)
    print(f"Total: {processed:,} mappings in {batch_idx} batches, {elapsed:.1f}s")
    if schedulable:
        print(f"SCHEDULABLE solution found at mapping {processed:,}")
    else:
        print(f"UNSCHEDULABLE — exhausted all {mapping_space_size:,} mappings")

    return system, schedulable, elapsed


def _single_priority_matrix(mapping_tuple, candidate_priorities, n):
    """Duplicate of BruteForceFPMappingAssignment._single_priority_matrix."""
    mapping = np.array(mapping_tuple).reshape(-1, 1)
    prio = np.array(candidate_priorities).reshape(-1, 1)
    return (prio < prio.T) * (mapping == mapping.T)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Diagnose brute-force mapping search progress")
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-u", "--utilization", type=float, default=0.5)
    parser.add_argument("-b", "--batch-size", type=int, default=50,
                        help="Mappings per batch (default: 50)")
    parser.add_argument("--batch-size2", type=int, default=500,
                        help="Second run with different batch size for comparison")
    args = parser.parse_args()

    run_bf_diagnostic(seed=args.seed, utilization=args.utilization,
                      batch_size=args.batch_size)

    if args.batch_size2:
        print()
        print()
        run_bf_diagnostic(seed=args.seed, utilization=args.utilization,
                          batch_size=args.batch_size2)
