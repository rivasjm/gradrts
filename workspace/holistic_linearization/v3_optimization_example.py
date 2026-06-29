"""
End-to-end example: priority optimisation with V3 differentiable surrogate.

Architecture:
  - **Surrogate cost** (V3): used during optimisation for the stop condition
  - **Surrogate gradient** (V3 autograd): drives the parameter updates
  - **Validation** (Holistic FP): run AFTER optimisation to measure real quality

This keeps the surrogate self-consistent (same model for cost & gradient).
"""
import os
from random import Random

import numpy as np
import torch

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from examples.generator import set_utilization
from gradient_descent.cost_functions import CostFunction
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.interfaces import ParameterHandler, GradientFunction
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from model.linear_system import LinearSystem, SchedulerType
from model.analysis_function import normalize_priorities
from model.linear_system_utils import backup_assignment, restore_assignment

from workspace.holistic_linearization.differentiable_v3 import v3_soft_priority

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Parameter handler:  raw priority values, clamped ≥ 0
# ---------------------------------------------------------------------------

class RawPriorityHandler(ParameterHandler):
    def extract(self, system: LinearSystem) -> list[float]:
        return [float(t.priority) for t in system.tasks]

    def insert(self, system: LinearSystem, x: list[float]):
        for t, xi in zip(system.tasks, x):
            t.priority = max(float(xi), 0.0)


# ---------------------------------------------------------------------------
# V3 surrogate cost  (differentiable model — used for stop condition)
# ---------------------------------------------------------------------------

class V3SoftCost(CostFunction):
    """Cost = smoothed invslack computed by the V3 soft-priority model."""

    def __init__(self, param_handler: ParameterHandler, tau=0.5):
        self.param_handler = param_handler
        self.tau = tau

    def compute(self, system: LinearSystem, x: list[float]) -> float:
        a = backup_assignment(system)
        self.param_handler.insert(system, x)

        tasks = system.tasks
        n = len(tasks)
        t2i = {t: i for i, t in enumerate(tasks)}

        pred_idx = [-1] * n
        for i, t in enumerate(tasks):
            p = t.predecessors
            pred_idx[i] = t2i[p[0]] if p else -1

        same_proc = torch.zeros(n, n, dtype=torch.bool)
        for i, ti in enumerate(tasks):
            for j, tj in enumerate(tasks):
                same_proc[i, j] = (ti.processor == tj.processor)

        C = torch.tensor([t.wcet for t in tasks], dtype=torch.float64)
        T = torch.tensor([t.period for t in tasks], dtype=torch.float64)

        try:
            s_init = torch.tensor(x, dtype=torch.float64)
            r, _s = v3_soft_priority(C, T, pred_idx, same_proc, tau=self.tau, s_init=s_init)
        except Exception:
            restore_assignment(system, a)
            return float('inf')

        last_idx = [i for i, t in enumerate(tasks) if t.is_last]
        deadlines = torch.tensor([tasks[i].flow.deadline for i in last_idx],
                                 dtype=torch.float64)

        slack = (r[last_idx] - deadlines) / deadlines
        soft_invslack = float(torch.logsumexp(slack * 5.0, dim=0) / 5.0)

        restore_assignment(system, a)
        return soft_invslack


# ---------------------------------------------------------------------------
# V3 autograd gradient
# ---------------------------------------------------------------------------

class V3SoftGradient(GradientFunction):
    """Gradient of avg-WCRT w.r.t. priorities via V3 autograd."""

    def __init__(self, tau=0.5):
        self.tau = tau

    def compute(self, system: LinearSystem, x: list[float]) -> list[float]:
        tasks = system.tasks
        n = len(tasks)
        t2i = {t: i for i, t in enumerate(tasks)}

        pred_idx = [-1] * n
        for i, t in enumerate(tasks):
            p = t.predecessors
            pred_idx[i] = t2i[p[0]] if p else -1

        same_proc = torch.zeros(n, n, dtype=torch.bool)
        for i, ti in enumerate(tasks):
            for j, tj in enumerate(tasks):
                same_proc[i, j] = (ti.processor == tj.processor)

        C = torch.tensor([t.wcet for t in tasks], dtype=torch.float64)
        T = torch.tensor([t.period for t in tasks], dtype=torch.float64)

        try:
            s_init = torch.tensor(x, dtype=torch.float64)
            r, s_leaf = v3_soft_priority(C, T, pred_idx, same_proc, tau=self.tau, s_init=s_init)
        except Exception:
            return [0.0] * n

        last_idx = [i for i, t in enumerate(tasks) if t.is_last]
        avg = r[last_idx].mean()
        avg.backward()

        g = s_leaf.grad
        if g is None:
            return [0.0] * n

        grad = g.detach().numpy()
        if np.any(~np.isfinite(grad)):
            return [0.0] * n

        # Rescale: limit gradient magnitude to avoid exploding steps
        gn = np.linalg.norm(grad)
        if gn > 1.0:
            grad = grad / gn

        return grad.tolist()


# ---------------------------------------------------------------------------
# Holistic FP evaluation
# ---------------------------------------------------------------------------

def holistic_cost(system: LinearSystem) -> float:
    HolisticFPAnalysis(limit_factor=1, reset=True).apply(system)
    wcrts = [f.wcrt for f in system.flows if f.wcrt is not None]
    deadlines = [f.deadline for f in system.flows]
    if not wcrts:
        return float('inf')
    return max((w - d) / d for w, d in zip(wcrts, deadlines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(seed=42, n_flows=3, n_tasks=4, n_procs=3, utilization=0.7,
        max_iter=200, lr=0.01):
    rnd = Random(seed)

    system = get_system(
        (n_flows, n_tasks, n_procs), rnd, name='opt_demo',
        deadline_factor_min=0.5, deadline_factor_max=1,
        sched=SchedulerType.FP, balanced=True,
    )
    set_utilization(system, utilization)

    print(f"System: {n_flows}x{n_tasks} tasks, {n_procs} procs, U={utilization:.2f}")

    # DM baseline
    pd = PDAssignment(normalize=True)
    pd.apply(system)
    cost_dm = holistic_cost(system)
    sched_dm = all(f.is_schedulable() for f in system.flows)
    print(f"DM baseline:  invslack={cost_dm:.4f}  sched={sched_dm}")

    # Optimizer: V3 surrogate (cost + gradient)
    param_handler = RawPriorityHandler()
    cost_function = V3SoftCost(param_handler=param_handler, tau=0.3)
    gradient_function = V3SoftGradient(tau=0.3)
    stop_function = ThresholdStopFunction(limit=max_iter, threshold=-0.05, patience=30)
    update_function = NoisyAdam(lr=lr, beta1=0.9, beta2=0.999, epsilon=0.01)

    optimizer = GradientDescentOptimizer(
        parameter_handler=param_handler,
        cost_function=cost_function,
        stop_function=stop_function,
        gradient_function=gradient_function,
        update_function=update_function,
        verbose=True,
    )

    print(f"\n--- Optimising (V3 surrogate, {max_iter} iter max, lr={lr}) ---")
    solution = optimizer.apply(system)

    # Validate with Holistic FP
    param_handler.insert(system, solution)
    normalize_priorities(system)

    cost_opt = holistic_cost(system)
    sched_opt = all(f.is_schedulable() for f in system.flows)
    delta = cost_dm - cost_opt
    print(f"\nOptimised:  invslack={cost_opt:.4f}  sched={sched_opt}")
    print(f"DM → Opt:   {cost_dm:.4f} → {cost_opt:.4f}  (delta={delta:+.4f})")

    return {
        'util': utilization,
        'cost_dm': cost_dm,
        'cost_opt': cost_opt,
        'sched_dm': sched_dm,
        'sched_opt': sched_opt,
        'improvement': cost_dm - cost_opt,
    }


if __name__ == '__main__':
    results = []
    for u in [0.55, 0.60, 0.65, 0.70, 0.75]:
        print("\n" + "=" * 60)
        r = run(seed=42, utilization=u, max_iter=100, lr=0.01)
        results.append(r)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        imp = r['improvement']
        sched_change = f"{r['sched_dm']}→{r['sched_opt']}"
        print(f"U={r['util']:.2f}  DM={r['cost_dm']:.4f} → Opt={r['cost_opt']:.4f}  "
              f"Δ={imp:+.4f}  sched: {sched_change}")
