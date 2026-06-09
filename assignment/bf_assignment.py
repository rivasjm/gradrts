import itertools
import math
import numpy as np

from assignment.assignments import PDAssignment
from model.analysis_function import AnalysisFunction
from model.linear_system import LinearSystem
from utils.exec_time import ExecTime
from vector.vector_fp import VectorHolisticFPAnalysis


class BruteForceAssignment(AnalysisFunction):
    def __init__(self, batch_size=10000, verbose=False):
        self.batch_size = batch_size if batch_size > 0 else 1
        self.verbose = verbose
        self.analysis = VectorHolisticFPAnalysis(limit_factor=10)
        self.schedulable = False
        self.exec_time = ExecTime()
        self.iterations_to_sched = -1
        self.space_size = 0

    def apply(self, system: LinearSystem) -> LinearSystem:
        self.exec_time.init()
        self.schedulable = False
        self.iterations_to_sched = -1

        pd = PDAssignment(normalize=True)
        pd.apply(system)

        tasks = system.tasks
        procs = system.processors

        proc_tasks_list = [proc.tasks for proc in procs]
        task_mapping = self._build_task_mapping(tasks, proc_tasks_list)

        prio_ranges = [range(1, len(p) + 1) for p in proc_tasks_list]
        prios = [list(itertools.permutations(p)) for p in prio_ranges]

        self.space_size = self._space_size(system)
        space = itertools.product(*prios)
        batch = []
        processed = 0

        for solution in space:
            trans = np.array(self._flatten(solution))[task_mapping].tolist()
            batch.append(trans)

            if len(batch) == self.batch_size:
                processed += len(batch)
                if self.verbose:
                    print(f"Processed {processed}/{self.space_size} "
                          f"({processed / self.space_size * 100:.3f}%)")
                if self._process_batch(system, batch):
                    self.schedulable = True
                    self.iterations_to_sched = processed
                    if self.verbose:
                        print("Schedulable solution found")
                    break
                batch.clear()

        if len(batch) > 0 and not self.schedulable:
            if self._process_batch(system, batch):
                self.schedulable = True
                self.iterations_to_sched = processed + len(batch)
                if self.verbose:
                    print("Schedulable solution found")

        self.exec_time.stop()
        return system

    def _process_batch(self, system, batch):
        scenarios = np.array(batch).T
        pm = self._build_priority_matrix(system, scenarios)
        self.analysis.apply(system, scenarios=pm)
        r = self.analysis.scenarios_response_times
        n = len(system.tasks)
        deadlines = np.array([task.flow.deadline for task in system.tasks]).reshape(n, 1)
        slacks = deadlines - r
        schedulables = np.all(slacks >= 0, axis=0)
        if np.any(schedulables):
            index = np.argmax(schedulables)
            solution = scenarios[:, index]
            for prio, task in zip(solution, system.tasks):
                task.priority = float(prio)
            return True
        return False

    def _build_priority_matrix(self, system, scenarios):
        n = len(system.tasks)
        s = scenarios.shape[1]
        procs = system.processors
        mapping = np.array([procs.index(task.processor) for task in system.tasks])
        mapping = mapping.reshape(1, n, 1)

        priorities = scenarios.T.reshape(s, n, 1)
        pm = (priorities.transpose(0, 2, 1) > priorities) & \
             (mapping == mapping.transpose(0, 2, 1)) & \
             ~np.eye(n, dtype=np.bool_).reshape(1, n, n)
        return pm.astype(np.float32)

    @staticmethod
    def _build_task_mapping(tasks, proc_tasks_list):
        proc_tasks_flat = [t for sublist in proc_tasks_list for t in sublist]
        return [proc_tasks_flat.index(task) for task in tasks]

    @staticmethod
    def _space_size(system):
        facts = [math.factorial(len(proc.tasks)) for proc in system.processors]
        return math.prod(facts)

    @staticmethod
    def _flatten(lst):
        return [x for sublist in lst for x in sublist]
