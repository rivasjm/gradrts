import math

from model.analysis_function import AnalysisFunction, reset_wcrt, init_wcrt, higher_priority
from model.linear_system import LinearSystem


class HolisticFPAnalysis(AnalysisFunction):
    def __init__(self, limit_factor=10, reset=False, verbose=False):
        self.limit_factor = limit_factor
        self.reset = reset
        self.verbose = verbose

    def reset_wcrts(self, system: LinearSystem):
        if self.reset:
            reset_wcrt(system)

    def apply(self, system: LinearSystem) -> None:
        init_wcrt(system)

        wcrts = [t.wcrt for t in system.tasks]
        wcrts_prev = [0 for t in system.tasks]

        while wcrts != wcrts_prev:  # LOOP wcrt convergence loop
            wcrts_prev = wcrts[:]

            for task in system.tasks:  # LOOP task loop
                hp = higher_priority(task)  # tasks of higher priority than 'task'
                limit = task.flow.deadline * self.limit_factor  # r limit for 'task'

                p = 1
                while True:  # LOOP p loop
                    w_prev = 0
                    w = p * task.wcet

                    while w != w_prev:  # LOOP w convergence loop
                        w_prev = w
                        w = sum(map(lambda t: math.ceil((t.jitter + w) / t.period) * t.wcet, hp)) + p * task.wcet
                        r = w - (p - 1) * task.period + task.jitter

                        if self.verbose:
                            print(f"{task.name} p={p} w={w:.3f} wprev={w_prev:.3f} r={r:.3f} wcrt={task.wcrt:.3f}")
                        if r > task.wcrt:
                            task.wcrt = r
                        if r > limit:
                            if self.reset:
                                self.reset_wcrts(system)
                            else:
                                for t in task.all_successors:
                                    t.wcrt = task.wcrt
                            return

                    if w <= p * task.period:
                        break  # no need to try more p's

                    p += 1

                # no need to do anything here, just jump to next task

            # retrieve the wcrts to see if they have changed since the last iteration
            wcrts = [t.wcrt for t in system.tasks]
