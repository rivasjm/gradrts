"""Equivalence test: sequential V1 (v1_analyse) vs VectorLinearizedV1Analysis.

Confirms the vectorised V1 implementation produces the same per-task WCRTs
as the sequential reference, on randomly generated systems.  Also benchmarks
the speed-up of batched evaluation vs sequential.
"""
import os
import time
from random import Random

import numpy as np

from examples.example_models import get_system
from model.analysis_function import init_wcrt, reset_wcrt
from model.linear_system import SchedulerType
from assignment.assignments import PDAssignment

from workspace.holistic_linearization.linearized_fp import LinearizedFPAnalysis
from workspace.holistic_linearization.vector_v1 import VectorLinearizedV1Analysis
from vector.vector_fp import PrioritiesMatrix


def run_sequential(system):
    reset_wcrt(system)
    init_wcrt(system)
    LinearizedFPAnalysis(limit_factor=10, max_p=100).apply(system)
    return [float(t.wcrt) for t in system.tasks]


def run_vector(system, scenarios=None):
    reset_wcrt(system)
    init_wcrt(system)
    v = VectorLinearizedV1Analysis(limit_factor=10, max_p=1000)
    v.apply(system, scenarios=scenarios)
    return v.full_response_times


def copy_wcrt_to(system):
    """Snapshot WCRTs so we can compare against an ext ref."""
    return {t.name: t.wcrt for t in system.tasks}


def main():
    rnd = Random(42)
    size = (3, 4, 3)
    n_systems = 50

    print(f"Generating {n_systems} systems {size} ...")
    systems = [
        get_system(size, rnd, name=str(i),
                   deadline_factor_min=0.5, deadline_factor_max=1,
                   sched=SchedulerType.FP, balanced=True)
        for i in range(n_systems)
    ]
    for s in systems:
        PDAssignment(normalize=True).apply(s)

    # 1) single-scenario (just the input system) equivalence
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    mismatches = 0
    for i, s in enumerate(systems):
        ref = run_sequential(s)
        # snapshot reference wcrt (sequential mutates the system)
        ref_wcrts = list(ref)
        # vectorised: reset, init, apply
        vec_r = run_vector(s)  # r is (t, 1)
        vec_wcrts = list(vec_r[:, 0])
        diff = np.array(vec_wcrts) - np.array(ref_wcrts)
        abs_diff = float(np.max(np.abs(diff))) if len(diff) else 0.0
        denom = np.maximum(np.abs(ref_wcrts), 1e-9)
        rel_diff = float(np.max(np.abs(diff) / denom)) if len(diff) else 0.0
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)
        # tolerate small float divergence from Jacobi vs Gauss-Seidel
        if rel_diff > 1e-3 or abs_diff > 0.5:
            mismatches += 1
            print(f"  sys{i}: abs={abs_diff:.4e} rel={rel_diff:.4e}\n"
                  f"   ref ={ref_wcrts}\n   vec ={vec_wcrts}")

    print(f"\n[single-system equivalence] n={n_systems}  "
          f"max_abs_diff={max_abs_diff:.4e}  "
          f"max_rel_diff={max_rel_diff:.4e}  mismatches={mismatches}")

    # 2) multi-scenario batching vs sequential
    print("\n[batch benchmark] comparing sequential N x V1 vs vectorised batch")
    builder = PrioritiesMatrix()
    batch_sizes = [8, 16, 24]
    for bsz in batch_sizes:
        # build random priority scenarios perturbed around the system's own
        n = len(systems[0].tasks)
        rng = np.random.default_rng(123)
        # one random input vector per scenario per system  (bsz scenarios)
        results_seq = []
        results_vec = []

        # Reference run: many sequential V1 calls
        # Build scenarios by perturbing the current priorities by small noise
        t0 = time.perf_counter()
        for s in systems:
            base_x = [float(t.priority) for t in s.tasks]
            scen_outputs = []
            for k in range(bsz):
                perturbed = base_x + rng.normal(0, 0.05, size=n)
                # apply priorities
                for ti, v in zip(s.tasks, perturbed):
                    ti.priority = float(max(v, 0.0))
                wcrts = run_sequential(s)
                scen_outputs.append(wcrts)
            results_seq.append(scen_outputs)
        t_seq = time.perf_counter() - t0

        # Restore base priorities and run vectorised batch
        for s in systems:
            PDAssignment(normalize=True).apply(s)
        # generate perturbed priority vectors as 'inputs'
        rng = np.random.default_rng(123)  # same seed -> same scenarios
        per_system_scenarios = []
        for s in systems:
            base_x = [float(t.priority) for t in s.tasks]
            arr = np.array([
                np.array(base_x) + rng.normal(0, 0.05, size=n)
                for _ in range(bsz)
            ], dtype=object)  # each row is a list-like priority vector
            per_system_scenarios.append(arr)
        # vector
        t0 = time.perf_counter()
        vec_full = []
        for s, arr in zip(systems, per_system_scenarios):
            scen_inputs = [list(r) for r in arr]
            scen_pm = builder.apply(s, scen_inputs)
            v = VectorLinearizedV1Analysis(limit_factor=10, max_p=1000)
            reset_wcrt(s)
            init_wcrt(s)
            v.apply(s, scenarios=scen_pm)
            # v.scenarios_response_times has shape (t, bsz)
            vec_full.append(v.scenarios_response_times.copy())
        t_vec = time.perf_counter() - t0

        # Compare equality between seq outputs and vec outputs
        diff_max = 0.0
        for s_idx, (seq_scen, vec_rt) in enumerate(zip(results_seq, vec_full)):
            for k in range(bsz):
                seq_vec = np.array(seq_scen[k])
                vec_col = vec_rt[:, k]
                d = float(np.max(np.abs(vec_col - seq_vec)))
                diff_max = max(diff_max, d)

        total_evals = n_systems * bsz
        print(f"  batch={bsz:3d}  seq={t_seq:7.2f}s  vec={t_vec:7.2f}s  "
              f"speedup={t_seq/max(t_vec,1e-9):5.1f}x  "
              f"max_diff={diff_max:.4e}")


if __name__ == "__main__":
    main()