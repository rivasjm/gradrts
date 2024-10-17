import math

from model.analysis_function import reset_wcrt, init_wcrt, LimitFactorReachedException
from model.linear_system import Task, Processor, LinearSystem, is_scheduler_type, SchedulerType


class HolisticGlobalEDFAnalysis:
    def __init__(self, limit_factor=10, reset=True, verbose=False):
        self.limit_factor = limit_factor
        self.reset = reset
        self.verbose = verbose

    @staticmethod
    def _activations(task: Task, length: float) -> int:
        """eq (4)"""
        return math.ceil((length+task.jitter)/task.period)

    def _longest_busy_period(self, proc: Processor, l_prev: float) -> float:
        length = sum(map(lambda t: math.ceil((l_prev+t.jitter)/t.period)*t.wcet, proc.tasks))
        if math.isclose(length, l_prev):
            return length
        else:
            return self._longest_busy_period(proc, length)

    def _set_psi(self, proc: Processor, busy_period: float):
        psi = [(p-1)*task.period - task.jitter + task.deadline
               for task in proc.tasks for p in range(1, self._activations(task, busy_period)+1)]
        return psi

    def apply(self, system: LinearSystem) -> None:
        if not is_scheduler_type(system, SchedulerType.EDF):
            print("System is not EDF")
            reset_wcrt(system)
            return

        init_wcrt(system)
        try:
            while True:
                changed = False
                for proc in system.processors:
                    changed |= self._proc_analysis(proc)
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

    def _proc_analysis(self, proc: Processor):
        length = self._longest_busy_period(proc, 0)
        changed = False
        for task in proc.tasks:
            changed |= self._task_analysis(task, length)
        return changed

    def _task_analysis(self, task: Task, length: float) -> bool:
        max_r = 0
        all_psi = self._set_psi(task.processor, length)
        for p in range(1, self._activations(task, length) + 1):
            activations = [psi - (p-1) * task.period + task.jitter - task.deadline for psi in all_psi
                           if (p-1)*task.period-task.jitter+task.deadline <= psi < p*task.period-task.jitter+task.deadline]
            for activation in activations:
                r = self._ra(task, activation, p)
                if r > max_r:
                    max_r = r
                if r > task.flow.deadline * self.limit_factor:
                    raise LimitFactorReachedException(task, r, task.flow.deadline * self.limit_factor)

        if max_r > task.wcrt:
            task.wcrt = max_r
            return True
        else:
            return False

    def _ra(self, task, activation, p):
        deadline_activation = activation - task.jitter + (p-1)*task.period + task.deadline
        wa = self._wa(task, deadline_activation, p, 0)
        ra = wa - activation + task.jitter - (p-1)*task.period
        return ra

    def _wa(self, task, deadline_activation, p, wa_prev):
        wa = p*task.wcet + sum(map(lambda t: self._wi(t, wa_prev, deadline_activation),
                                   [t for t in task.processor.tasks if t != task]))
        if math.isclose(wa, wa_prev):
            return wa
        else:
            return self._wa(task, deadline_activation, p, wa)

    @staticmethod
    def _wi(task: Task, t: float, D: float) -> float:
        value = min(math.ceil((t+task.jitter)/task.period), math.floor((task.jitter + D - task.deadline)/task.period)+1)
        return value * task.wcet if value > 0 else 0