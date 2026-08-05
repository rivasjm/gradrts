from model.analysis_function import reset_wcrt
from examples.generator import set_utilization
from model.linear_system import LinearSystem
import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from datetime import datetime
from model.linear_system_utils import backup_assignment, restore_assignment


# ANSI escape codes for colors
RED = '\033[91m'
RESET = '\033[0m'

METHOD_COLORS = {
    "gdpa": "#1f77b4",
    "gdpa+map": "#ff7f0e",
    "hopa": "#8c564b",
    "pd": "#17becf",
    "bf": "#d62728",
}


def _display_method_label(label):
    """Normalize evaluation labels for the efficiency chart."""
    normalized = label.lower()
    if "mapping" in normalized:
        return "gdpa+map"
    if "gdpa" in normalized:
        return "gdpa"
    if "hopa" in normalized:
        return "hopa"
    if "pd" in normalized:
        return "pd"
    if "bf" in normalized:
        return "bf"
    return label


def _scenario_label(name):
    normalized = name.lower()
    if "mapping" in normalized:
        return "MAP"
    if "edf" in normalized:
        return "EDF"
    return "FP"

class SchedRatioEval:
    """Evaluate schedulability ratios of multiple methods across a range of utilizations.

    For each utilization level, sets all systems to that utilization, then runs each
    method on every system in parallel. Produces line/bar charts (PNG) and spreadsheets
    (XLSX) of schedulability ratios and average execution times.

    Parameters
    ----------
    name : str
        Study name, used as prefix for output files.
    labels : list of str
        Display names for each method.
    funcs : list of callable
        Functions ``f(system) -> bool``, one per label.
    systems : list of LinearSystem
        Population of systems to evaluate.
    utilizations : array-like
        Utilization levels to sweep (e.g., ``np.linspace(0.5, 0.9, 20)``).
    threads : int
        Number of worker processes.
    preprocessor : callable, optional
        Applied to each system before analysis.
    utilization_func : callable, optional
        Sets utilization on a system. Default: ``set_utilization(system, u)``.
    output_dir : str, optional
        Directory for output files. Default: current working directory.
    show : bool, default False
        If True, display live-updating charts during the sweep (non-blocking).
    """
    def __init__(self, name, labels, funcs, systems, utilizations, threads,
                 preprocessor=None, utilization_func=set_utilization,
                 output_dir=None, show=False):
        assert len(labels) == len(funcs)
        self.name = name
        self.labels = labels
        self.funcs = funcs
        self.systems = systems
        self.utilizations = utilizations
        self.threads = threads
        self.preprocessor = preprocessor
        self.utilization_func = utilization_func
        self.start = None
        self.output_dir = output_dir or os.getcwd()
        self.show = show
        self._figs = {}

    def run(self):
        """Run the full evaluation sweep. Generates PNG and XLSX files in output_dir.

        If show=True, live-updating charts are displayed without blocking."""
        self.start = time.time()
        if self.show:
            plt.ion()
        job = 0
        all_results = np.zeros((len(self.utilizations), len(self.labels)))
        all_times = np.zeros((len(self.utilizations), len(self.labels)))

        for u_index, u in enumerate(self.utilizations):
            for s in self.systems:
                self.utilization_func(s, u)
                if self.preprocessor:
                    self.preprocessor(s)

            with Pool(self.threads) as pool:
                f = partial(self._step, u_index=u_index)
                for scheds, times in pool.imap_unordered(f, self.systems):
                    job += 1
                    all_results[u_index, :] += scheds
                    all_times[u_index, :] += times
                    print(f"{datetime.now()} : u={u} job={job}")

            self._save(all_results, "schedulables", self.show)
            self._save(all_times / len(self.systems), "times", False)

        # Aggregate efficiency scatter (total schedulable vs total time)
        self._efficiency_chart(all_results, all_times)

        if self.show:
            plt.ioff()
            plt.close("all")

    def _step(self, system: LinearSystem, u_index: int):
        """Make sure I leave the system in the same state as before"""
        results = np.zeros(len(self.funcs), dtype=np.int8)
        times = np.zeros(len(self.funcs), dtype=np.single)
        a = backup_assignment(system)
        for f, func in enumerate(self.funcs):
            try:
                reset_wcrt(system)
                before = time.perf_counter()
                sched = func(system)
                after = time.perf_counter()
                restore_assignment(system, a)
                if sched:
                    results[f] = 1
                times[f] = after - before
            except Exception as e:
                print(f"{RED}Error in {self.labels[f]}, system={system.name}\n{e}{RESET}")
                restore_assignment(system, a)
                results[f] = 0
                times[f] = 0
        return results, times

    def _save(self, data, suffix, show):
        label = f"{self.name}_{suffix}"
        length = len(self.utilizations)
        if length > 1:
            self._line_chart(label, data, ylabel=suffix, save=True, show=show)
        self._bar_chart(label, data, ylabel=suffix, save=True, show=length == 1 and show)
        self._excel(label, data)

    def _path(self, filename):
        return os.path.join(self.output_dir, filename)

    def _line_chart(self, label, data, ylabel, save=True, show=True):
        plt.clf()
        df = pd.DataFrame(data=data,
                          index=self.utilizations,
                          columns=self.labels)
        fig, ax = plt.subplots()
        df.plot(ax=ax)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Average utilization")

        # print system size
        ax.annotate(self.name, xy=(0, -0.1), xycoords='axes fraction', ha='left', va="center", fontsize=8)

        # register execution vector_times
        time_label = f"{time.time() - self.start:.2f} seconds"
        ax.annotate(time_label, xy=(1, -0.1), xycoords='axes fraction', ha='right', va="center", fontsize=8)
        fig.tight_layout()
        if save:
            fig.savefig(self._path(f"{label}.png"))
        if show:
            plt.show()
        plt.close(fig)

    def _bar_chart(self, label, data, ylabel, save=True, show=True):
        plt.clf()
        df = pd.DataFrame(data=data, columns=self.labels)
        fig, ax = plt.subplots()
        df.sum().plot.barh(ax=ax)
        ax.tick_params(axis='both', which='major', labelsize=6)

        # print system size
        ax.annotate(self.name, xy=(0, -0.1), xycoords='axes fraction', ha='left', va="center", fontsize=8)

        # register execution vector_times
        time_label = f"{time.time() - self.start:.2f} seconds"
        ax.annotate(time_label, xy=(1, -0.1), xycoords='axes fraction', ha='right', va="center", fontsize=8)
        fig.tight_layout()
        if save:
            fig.savefig(self._path(f"{label}_summary.png"))
        if show:
            plt.show()
        plt.close(fig)

    def _excel(self, label, data):
        df = pd.DataFrame(data=data,
                          index=self.utilizations,
                          columns=self.labels)
        df.to_excel(self._path(f"{label}.xlsx"))

    def _efficiency_chart(self, results, times):
        """Scatter plot: total schedulable vs total execution time per method."""
        total_sched = results.sum(axis=0)
        total_time = times.sum(axis=0)
        n_systems = len(self.systems) * len(self.utilizations)

        plt.clf()
        fig, ax = plt.subplots(figsize=(7, 5.5))
        labels = [_display_method_label(label) for label in self.labels]

        for i, label in enumerate(labels):
            color = METHOD_COLORS.get(label, plt.cm.tab10(i))
            ax.scatter(total_time[i], total_sched[i],
                       s=180, color=color, edgecolors='white',
                       linewidth=1.5, zorder=5)
            is_pd = label == "pd"
            ax.annotate(
                label,
                (total_time[i], total_sched[i]),
                textcoords="offset points",
                xytext=(8 if is_pd else -12, 6),
                fontsize=18,
                fontweight="bold",
                color=color,
                ha="left" if is_pd else "right",
            )

        ax.set_xscale("log")
        ax.set_ylim(0, max(1000, n_systems))
        ax.set_xlabel("Total execution time (s)", fontweight="bold", fontsize=18)
        ax.set_ylabel("Schedulable systems (/1000)", fontweight="bold", fontsize=18)
        ax.grid(True, which="both", axis="both", alpha=0.3)
        ax.tick_params(axis="both", labelsize=16)
        ax.text(
            0.95,
            0.05,
            _scenario_label(self.name),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontweight="bold",
            fontsize=18,
            bbox={"boxstyle": "round", "ec": "black", "fc": "bisque"},
        )

        fig.tight_layout()
        fig.savefig(self._path(f"{self.name}_efficiency.png"))
        if self.show:
            plt.show()
        plt.close(fig)
