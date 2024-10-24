import math

from model.analysis_function import reset_wcrt, init_wcrt, LimitFactorReachedException, AnalysisFunction
from model.linear_system import Task, LinearSystem, SchedulerType, is_scheduler_type


class HolisticLocalEDFAnalysis(AnalysisFunction):
    """
    Based on paper "Optimized Deadline Assignment and schedulability Analysis for Distributed Real-Time Systems
    with Local EDF Scheduling
    """

    def __init__(self, limit_factor=10, reset=True, verbose=False):
        self.limit_factor = limit_factor
        self.reset = reset
        self.verbose = verbose

    @staticmethod
    def _wi(task: Task, t: float, D: float) -> float:
        """Eq (1)"""
        pl = math.ceil((t + task.jitter) / task.period)
        pd = 0 if D < task.deadline else math.floor((task.jitter + D - task.deadline) / task.period) + 1
        m = min(pl, pd)
        return m * task.wcet if m > 0 else 0

    @classmethod
    def _busy_period(cls, task: Task, l_prev: float) -> float:
        """Eq (5) [adapted from Mast implementation]"""
        own = math.ceil(l_prev / task.period) * task.wcet
        tasks = [t for t in task.processor.tasks if t != task]
        length = own + sum(map(lambda t: math.ceil((l_prev + t.jitter) / t.period) * t.wcet, tasks))
        if math.isclose(length, l_prev):
            return length
        else:
            return cls._busy_period(task, length)

    @classmethod
    def _build_set_psi(cls, task: Task, busy_period: float, p: int):
        # eq (4) [adapted from Mast implementation]
        tasks = [t for t in task.processor.tasks if t != task]
        psi_ij = {(p - 1) * t.period - t.jitter + t.deadline
                  for t in tasks for p in range(1, math.ceil((busy_period + t.jitter) / t.period) + 1)
                  if (p - 1) * t.period - t.jitter >= 0}
        psi_ij |= {t.deadline for t in tasks}

        # Eq (6)
        psi_ab = {(p - 1) * task.period + task.deadline
                  for p in range(1, math.ceil(busy_period / task.period) + 1)}

        set_psi = psi_ij | psi_ab
        return set_psi

    @staticmethod
    def _ra(task, psi, wab):
        """Eq (9)"""
        rab = wab - psi + task.deadline + task.jitter
        return rab

    @classmethod
    def _wab(cls, task, psi, p, wab_prev):
        """Eq (8)"""
        wab = p * task.wcet + sum(map(lambda t: cls._wi(t, wab_prev, psi),
                                      [t for t in task.processor.tasks if t != task]))
        if math.isclose(wab, wab_prev):
            return wab
        else:
            return cls._wab(task, psi, p, wab)

    def apply(self, system: LinearSystem) -> None:
        if not is_scheduler_type(system, SchedulerType.EDF):
            print("System is not EDF")
            reset_wcrt(system)
            return

        init_wcrt(system)
        try:
            while True:
                changed = False
                for task in system.tasks:
                    changed |= self._task_analysis(task)
                if not changed:
                    break
        except LimitFactorReachedException as e:
            if self.verbose:
                print(e.message)
            if self.reset:
                reset_wcrt(system)
            else:
                e.task.wcrt = e.response_time
                for task in e.task.all_successors:
                    task.wcrt = e.response_time

    def _task_analysis(self, task: Task) -> bool:
        """task: task under analysis"""
        length = self._busy_period(task, task.wcet)
        max_r = 0
        for p in range(1, math.ceil(length / task.period) + 1):
            psi_set = {psi for psi in self._build_set_psi(task, length, p) if
                       (p - 1) * task.period + task.deadline <= psi < p * task.period + task.deadline}
            for psi in psi_set:
                w = self._wab(task, psi, p, p * task.wcet)  # converges to a w value
                r = self._ra(task, psi, w)
                if r > max_r:
                    max_r = r
                if r > task.flow.deadline * self.limit_factor:
                    raise LimitFactorReachedException(task, r, task.flow.deadline * self.limit_factor)

        if max_r > task.wcrt:
            task.wcrt = max_r
            return True
        else:
            return False
