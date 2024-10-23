from model.analysis_function import reset_wcrt
from examples.generator import set_utilization
from model.linear_system import LinearSystem
import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
from model.linear_system_utils import backup_assignment, restore_assignment

class SchedRatioEval:
    """Class to perform a Schedulability Ratio evaluation over a utilization series"""
    def __init__(self, name, labels, funcs, systems, utilizations, threads,
                 preprocessor=None, utilization_func=set_utilization):
        assert len(labels) == len(funcs)
        self.name = name                    # name of the study. used to name output files
        self.labels = labels                # label for each function (same length as funcs)
        self.funcs = funcs                  # test functions: each function maps system -> true/false
        self.systems = systems              # population of systems
        self.utilizations = utilizations    # utilizations array (usually: each number between [0,1], and increasing)
        self.threads = threads              # number of CPU threads to use
        self.preprocessor = preprocessor    # function to pre-process system before analyzing it (optional)
        self.utilization_func = utilization_func  # function to set the system utilization
        self.start = None                   # starting time

    def run(self):
        self.start = time.time()
        job = 0
        results = np.zeros((len(self.utilizations), len(self.labels)))          # schedulability ratio
        running_times = np.zeros((len(self.utilizations), len(self.labels)))    # execution times

        for u_index, u in enumerate(self.utilizations):
            # set utilization to every system
            for s in self.systems:
                self.utilization_func(s, u)

            # finish all the executions for one utilization before advancing to the next utilization
            # use a thread pool to accelerate the execution
            # each thread analyzes each system (no 2 threads analyze the same system at the same time)
            with Pool(self.threads) as pool:
                f = partial(self._step, u_index=u_index)
                for scheds, times in pool.imap_unordered(f, self.systems):
                    job += 1
                    results[u_index, :] += scheds
                    running_times[u_index, :] += times
                    print(f"{datetime.now()} : u={u} job={job}")

            # update results file
            self._save(results, "schedulables")
            self._save(running_times/len(self.systems), "times", show=False)

    def _step(self, system: LinearSystem, u_index: int):
        """Make sure I leave the system in the same state as before"""
        results = np.zeros(len(self.funcs), dtype=np.int8)
        times = np.zeros(len(self.funcs), dtype=np.single)
        a = backup_assignment(system)
        for f, func in enumerate(self.funcs):
            if self.preprocessor:
                self.preprocessor(system)
            reset_wcrt(system)
            before = time.perf_counter()
            sched = func(system)
            after = time.perf_counter()
            restore_assignment(system, a)
            if sched:
                results[f] = 1
            times[f] = after - before
        return results, times

    def _save(self, data, suffix, show=True):
        label = f"{self.name}_{suffix}"
        length = len(self.utilizations)
        if length > 1:
            self._line_chart(label, data, ylabel=suffix, save=True, show=show)
        self._bar_chart(label, data, ylabel=suffix, save=True, show=length == 1 and show)
        self._excel(label, data)

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
            fig.savefig(f"{label}.png")
        if show:
            plt.show()

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
            fig.savefig(f"{label}_summary.png")
        if show:
            plt.show()

    def _excel(self, label, data):
        df = pd.DataFrame(data=data,
                          index=self.utilizations,
                          columns=self.labels)
        df.to_excel(f"{label}.xlsx")
