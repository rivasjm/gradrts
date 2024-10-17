import sys

from assignment.pd_assignment import PDAssignment
from model.analysis_function import globalize_deadlines, extract_assignment, insert_assignment, calculate_priorities, \
    normalize_priorities, AnalysisFunction
from model.linear_system import LinearSystem, Task, Processor, Flow
from utils.exec_time import ExecTime


class HOPAssignment(AnalysisFunction):
    def __init__(self, analysis, iterations=40, k_pairs=None, patience=40, over_iterations=0,
                 callback=None, normalize=False, globalize=False, verbose=False):
        self.analysis = analysis
        self.k_pairs = k_pairs if k_pairs else HOPAssignment.default_k_pairs()
        self.iterations = iterations
        self.patience = patience
        self.over_iterations = over_iterations
        self.callback = callback
        self.globalize = globalize
        self.verbose = verbose
        self.normalize = normalize
        self.exec_time = ExecTime()
        self.iterations_to_sched = -1

    @staticmethod
    def default_k_pairs():
        return [(2.0, 2.0), (1.8, 1.8), (3.0, 3.0), (1.5, 1.5)]

    def apply(self, system: LinearSystem):
        self.exec_time.init()
        self.iterations_to_sched = -1
        iteration = 0
        patience = self.patience if self.patience >= 0 else 100
        over_iterations = self.over_iterations
        stop = False
        optimizing = False
        best_slack = float("-inf")

        PDAssignment.calculate_local_deadlines(system)
        if self.globalize:
            globalize_deadlines(system)
        best_assignment = extract_assignment(system)

        for ka, kr in self.k_pairs:
            insert_assignment(system, best_assignment)  # always start each new k-pair iteration with the best

            for i in range(self.iterations):
                iteration += 1
                if self.verbose:
                    print(f"Iteration={i}, ka={ka}, kr={kr} ", end="")

                changed = calculate_priorities(system)  # update priorities
                patience = patience-1 if not changed else self.patience

                system.apply(self.analysis)  # update response times
                self.clean_response_times(system)
                if self.callback:
                    self.callback.apply(system)

                slack = system.slack
                if slack > best_slack:
                    best_slack = slack
                    best_assignment = extract_assignment(system)

                if self.verbose:
                    sched = "SCHEDULABLE" if system.is_schedulable() else "NOT SCHEDULABLE"
                    print(f"slack={system.slack} {sched}")

                if system.is_schedulable() and self.iterations_to_sched < 0:
                    self.iterations_to_sched = iteration

                if system.is_schedulable() and over_iterations > 0:
                    optimizing = True

                if optimizing:
                    over_iterations -= 1

                if (not optimizing and system.is_schedulable()) or patience <= 0:
                    stop = True
                    break
                elif optimizing and over_iterations < 0 or patience <= 0:
                    stop = True
                    break

                self.update_local_deadlines(system, ka, kr)
                if self.globalize:
                    globalize_deadlines(system)

            if stop:
                break

        self.delete_excesses(system)
        insert_assignment(system, best_assignment)
        self.exec_time.stop()
        system.apply(self.analysis)
        if self.verbose:
            sched = "SCHEDULABLE" if system.is_schedulable() else "NOT SCHEDULABLE"
            print(f"Returning best assignment: slack={system.slack} {sched}")
        if self.normalize:
            normalize_priorities(system)

    def update_local_deadlines(self, system: LinearSystem, ka, kr):
        # update excesses with last response times
        for task in system.tasks: self.save_task_excess(task)
        for proc in system.processors: self.save_proc_excess(proc)
        for flow in system.flows: self.save_flow_mex(flow)
        self.save_proc_mex(system)

        # calculate unadjusted local deadlines
        for task in system.tasks:
            self.save_local_deadline(task, ka, kr)

        # adjust local deadlines
        self.adjust_local_deadlines(system)

    @staticmethod
    def save_local_deadline(task: Task, ka, kr):
        mex_pr = task.flow.system.mex_pr
        second = 1 + task.processor.excess/(kr * mex_pr) if kr * mex_pr != 0 else sys.float_info.max
        third = 1 + task.excess/(ka * task.flow.excess) if ka * task.flow.excess != 0 else sys.float_info.max
        task.deadline = task.deadline * second * third

    @staticmethod
    def save_task_excess(task: Task):
        d = task.deadline
        e = 0
        if d <= task.period:
            e = (task.wcrt-d)*task.flow.wcrt/task.flow.deadline
        elif d > task.period:
            e = (task.wcrt+task.jitter-d)*task.flow.wcrt/task.flow.deadline
        task.excess = e

    @staticmethod
    def save_proc_excess(proc: Processor):
        proc.excess = sum([task.excess for task in proc.tasks])

    @staticmethod
    def save_flow_mex(flow: Flow):
        excesses = [abs(task.excess) for task in flow.tasks]
        flow.excess = max(excesses) if len(excesses) > 0 else 0

    @staticmethod
    def save_proc_mex(system: LinearSystem):
        excesses = [abs(proc.excess) for proc in system.processors]
        system.mex_pr = max(excesses) if len(excesses) > 0 else 0

    @staticmethod
    def delete_excesses(system: LinearSystem):
        for task in system.tasks:
            if hasattr(task, "excess"): del task.excess
        for flow in system.flows:
            if hasattr(flow, "excess"): del flow.excess
        for proc in system.processors:
            if hasattr(proc, "excess"): del proc.excess
        if hasattr(system, "mex_pr"): del system.mex_pr

    @staticmethod
    def adjust_local_deadlines(system: LinearSystem):
        for flow in system.flows:
            d_sum = sum([task.deadline for task in flow])
            for task in flow:
                task.deadline = task.deadline * flow.deadline / d_sum

    @staticmethod
    def clean_response_times(system):
        for task in system.tasks:
            if task.wcrt is None:
                task.wcrt = sys.float_info.max