"""Instrumented holistic local-EDF analysis.

This is the *same* response-time computation as the judge in
``frozen/analysis.py``, but it also records **how** each response time is built:

- the busy period, and the **binding** activation index ``p*`` and ``psi*`` point
  (the pair that produced the maximum ``r_ab``);
- there, for every interfering task on the same processor: how many activations
  ``m`` it is credited with, whether the **deadline cap** of eq (1) (``pd``)
  binds instead of the activation bound (``pl``), and the interference
  ``m * C`` it contributes;
- whether the analysis aborted (a response above ``flow.deadline *
  limit_factor``).

This is the tool for **deriving** a deadline-assignment rule: it shows which
interferer's cap matters and where a local deadline actually enters.

    from frozen import explain
    schedulable, details = explain(system)          # limit_factor=1 (the judge)
    print(format_detail("a1", details["a1"]))
    schedulable, details = explain(system, limit_factor=1000)   # untruncated R
"""

import math

from .analysis import HolisticLocalEDFAnalysis, LimitFactorReachedException


class InstrumentedHolisticLocalEDFAnalysis(HolisticLocalEDFAnalysis):
    """``HolisticLocalEDFAnalysis`` that also records the interference breakdown.

    After ``apply``, ``self.details`` maps each task name to a dict with the
    busy period, the binding ``(p, psi, w)``, the per-interferer contributions
    and an ``aborted`` flag.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.details = {}

    def _breakdown(self, task, psi, w):
        contributions = {}
        for t in task.processor.tasks:
            if t is task:
                continue
            pl = math.ceil((w + t.jitter) / t.period)
            pd = 0 if psi < t.deadline else math.floor(
                (t.jitter + psi - t.deadline) / t.period) + 1
            m = min(pl, pd)
            contributions[t.name] = {
                "pl": pl, "pd": pd, "m": m, "capped": pd < pl,
                "deadline": t.deadline, "jitter": t.jitter, "period": t.period,
                "wcet": t.wcet, "interference": m * t.wcet,
            }
        return contributions

    def _task_analysis(self, task):
        if self._timeout():
            raise LimitFactorReachedException(task, task.wcrt or 0, float("inf"))
        length = self._busy_period(task, task.wcet)
        max_r = 0
        best = None
        psi_ij = self._build_set_psi_ij(task, length)
        for p in range(1, math.ceil(length / task.period) + 1):
            psi_ab = self._build_set_psi_ab(task, length, p)
            psi_set = {psi for psi in (psi_ij | psi_ab)
                       if (p - 1) * task.period + task.deadline <= psi
                       < p * task.period + task.deadline}
            for psi in psi_set:
                w = self._wab(task, psi, p, p * task.wcet)
                r = self._ra(task, psi, w)
                if r > max_r:
                    max_r = r
                    best = (p, psi, w)
                if self._timeout() or r > task.flow.deadline * self.limit_factor:
                    self._record(task, length, best, max_r, aborted=True)
                    raise LimitFactorReachedException(
                        task, max_r, task.flow.deadline * self.limit_factor)
        self._record(task, length, best, max_r, aborted=False)
        if max_r > task.wcrt:
            task.wcrt = max_r
            return True
        return False

    def _record(self, task, length, best, max_r, aborted):
        detail = {
            "busy_period": length,
            "deadline": task.deadline,
            "jitter": task.jitter,
            "period": task.period,
            "wcet": task.wcet,
            "response": max_r,
            "aborted": aborted,
        }
        if best is not None:
            p, psi, w = best
            detail.update({
                "binding_p": p,
                "binding_psi": psi,
                "binding_w": w,
                "contributions": self._breakdown(task, psi, w),
            })
        self.details[task.name] = detail


def explain(system, limit_factor=1, reset=True, verbose=False):
    """Analyse ``system`` and return ``(schedulable, details)``.

    ``limit_factor=1`` matches the judge (a response above ``flow.deadline``
    aborts and, with ``reset``, the system is unschedulable). Raise it to see
    the untruncated response times when a task overshoots its deadline.
    """
    analysis = InstrumentedHolisticLocalEDFAnalysis(
        limit_factor=limit_factor, reset=reset, verbose=verbose)
    analysis.apply(system)
    return system.is_schedulable(), analysis.details


def format_detail(name, detail):
    """A readable, multi-line rendering of one task's breakdown."""
    head = (f"{name}: R={detail['response']:.3f}  d={detail['deadline']:.3f}  "
            f"j={detail['jitter']:.3f}  busy={detail['busy_period']:.3f}"
            + ("   [ABORTED]" if detail["aborted"] else ""))
    lines = [head]
    if "contributions" in detail:
        lines.append(f"    binding: p={detail['binding_p']}  "
                     f"psi={detail['binding_psi']:.3f}  w={detail['binding_w']:.3f}")
        for cn, c in sorted(detail["contributions"].items(),
                            key=lambda kv: -kv[1]["interference"]):
            flag = "CAP" if c["capped"] else "   "
            lines.append(f"      {cn:14s} m={c['m']:<3d} pl={c['pl']:<3d} "
                         f"pd={c['pd']:<3d} [{flag}]  d={c['deadline']:9.3f}  "
                         f"I={c['interference']:.3f}")
    return "\n".join(lines)
