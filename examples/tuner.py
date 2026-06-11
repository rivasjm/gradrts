import itertools
import os
import time
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from model.analysis_function import reset_wcrt
from model.linear_system_utils import backup_assignment, restore_assignment

RED = '\033[91m'
RESET = '\033[0m'


class GradientHyperTuner:
    """Find optimal hyperparameters for a gradient descent configuration.

    For each combination in the cartesian product of ``param_grid``, constructs
    an optimizer via ``build_optimizer(**params)``, applies it to every system,
    and records schedulability and execution time.  Outputs a ranking XLSX and
    a top-N horizontal bar chart.

    Parameters
    ----------
    name : str
        Study name, used as prefix for output files.
    systems : list of LinearSystem
        Systems to evaluate.  Used as-is (no utilization or preprocessing applied).
    param_grid : dict
        Hyperparameter names mapped to lists of values to sweep.
        Example: ``{"lr": [0.5, 1.5], "sigma": [1.0, 3.0]}``
    build_optimizer : callable
        Factory ``(**params) -> Function`` that wires together all gradient-descent
        modules and returns a fully configured ``GradientDescentOptimizer``.
    threads : int
        Number of worker processes for parallel system evaluation.
    final_analysis : AnalysisFunction, optional
        Exact schedulability analysis run after the optimizer to compute accurate
        WCRTs.  Default: ``HolisticFPAnalysis(limit_factor=1, reset=True)``.
    output_dir : str, optional
        Directory for output files.  Default: current working directory.
    top_n : int, default 10
        Number of best combinations to show in the bar chart.
    verbose : bool, default False
        Print progress per combination.
    """

    def __init__(self, name, systems, param_grid, build_optimizer, threads,
                 final_analysis=None, output_dir=None, top_n=10, verbose=False):
        self.name = name
        self.systems = systems
        self.param_grid = param_grid
        self.build_optimizer = build_optimizer
        self.threads = threads
        self.final_analysis = final_analysis or HolisticFPAnalysis(limit_factor=1, reset=True)
        self.output_dir = output_dir or os.getcwd()
        self.top_n = top_n
        self.verbose = verbose

    def _print_banner(self, keys, values, n_combos, n_systems, total_evals):
        print(f"=== GradientHyperTuner: {self.name} ===")
        print(f"Systems:       {n_systems}")
        print(f"Param grid:    {len(keys)} parameters, {n_combos} combinations")
        for k, v in zip(keys, values):
            print(f"  {k:20s} {v}")
        print(f"Total evals:   {total_evals} ({n_combos} combos x {n_systems} systems)")
        print(f"Threads:       {self.threads}")
        print(f"Output:        {self.output_dir}")
        print(f"Top N:         {self.top_n}")
        print(f"{'=' * 50}")

    def run(self):
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = list(itertools.product(*values))

        n_systems = len(self.systems)
        n_combos = len(combinations)

        total_evals = n_combos * n_systems
        self._print_banner(keys, values, n_combos, n_systems, total_evals)

        schedulables = np.zeros(n_combos, dtype=np.int32)
        times = np.zeros(n_combos, dtype=np.float64)

        start = time.perf_counter()

        for idx, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            if self.verbose:
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"[{idx + 1}/{n_combos}] {params_str} ...", end=" ", flush=True)

            with Pool(self.threads) as pool:
                worker = partial(_tuner_worker, params=params,
                                 build_optimizer=self.build_optimizer,
                                 final_analysis=self.final_analysis)
                for sched, elapsed in pool.imap_unordered(worker, self.systems):
                    if sched:
                        schedulables[idx] += 1
                    times[idx] += elapsed

            ratio = schedulables[idx] / n_systems
            if self.verbose:
                print(f"-> {schedulables[idx]}/{n_systems} ({ratio:.1%})")
            else:
                params_str = ", ".join(f"{k}={v}" for k, v in params.items())
                print(f"[{idx + 1}/{n_combos}] {params_str} -> {schedulables[idx]}/{n_systems} ({ratio:.1%})")

        elapsed_total = time.perf_counter() - start
        times /= n_systems

        self._save_ranking(keys, combinations, schedulables, times)
        self._save_best(keys, combinations, schedulables)

        best_idx = int(np.argmax(schedulables))
        best_params = dict(zip(keys, combinations[best_idx]))
        best_ratio = schedulables[best_idx] / n_systems
        print(f"Best: {best_params} -> {schedulables[best_idx]}/{n_systems} ({best_ratio:.1%})")
        print(f"Total time: {elapsed_total:.1f}s")

    def _save_ranking(self, keys, combinations, schedulables, times):
        n_systems = len(self.systems)
        data = {}
        for col, vals in zip(keys, np.array(combinations).T):
            data[col] = vals
        data["schedulable"] = schedulables
        data["ratio"] = schedulables / n_systems
        data["avg_time_s"] = times.round(4)

        df = pd.DataFrame(data)
        df.sort_values("schedulable", ascending=False, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.index.name = "rank"

        path = os.path.join(self.output_dir, f"{self.name}_tuning.xlsx")
        df.to_excel(path)
        if self.verbose:
            print(f"Ranking saved to {path}")

    def _save_best(self, keys, combinations, schedulables):
        n_systems = len(self.systems)
        order = np.argsort(schedulables)[::-1]
        top = min(self.top_n, len(order))
        ratios = schedulables[order[:top]] / n_systems

        labels = []
        for i in range(top):
            params = dict(zip(keys, combinations[order[i]]))
            label = ", ".join(f"{k}={v}" for k, v in params.items())
            labels.append(label)

        fig, ax = plt.subplots()
        ax.barh(range(top), ratios[::-1], tick_label=labels[::-1])
        ax.set_xlabel("schedulability ratio")
        ax.set_title(f"{self.name} — top {top} hyperparameter combinations")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        path = os.path.join(self.output_dir, f"{self.name}_top{top}.png")
        fig.savefig(path)
        plt.close(fig)
        if self.verbose:
            print(f"Top-{top} chart saved to {path}")


def _tuner_worker(system, params, build_optimizer, final_analysis):
    optimizer = build_optimizer(**params)
    a = backup_assignment(system)
    try:
        reset_wcrt(system)
        before = time.perf_counter()
        if callable(optimizer):
            optimizer(system)
        else:
            optimizer.apply(system)
        final_analysis.apply(system)
        after = time.perf_counter()
        sched = system.is_schedulable()
        restore_assignment(system, a)
        return sched, after - before
    except Exception as e:
        restore_assignment(system, a)
        print(f"{RED}Error in tuner worker, system={system.name}, params={params}\n{e}{RESET}")
        return False, 0
