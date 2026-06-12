import argparse
import os
from datetime import datetime
from random import Random
import numpy as np

from analysis.holistic_fp_analysis import HolisticFPAnalysis
from assignment.assignments import PDAssignment
from examples.example_models import get_system
from examples.generator import unbalance_contended, set_utilization
from examples.tuner import GradientHyperTuner
from gradient_descent.gradient_optimizer import GradientDescentOptimizer
from gradient_descent.parameter_handlers import MappingOnlyExtractor
from gradient_descent.cost_functions import InvslackCost
from gradient_descent.stop_functions import ThresholdStopFunction
from gradient_descent.update_functions import NoisyAdam
from vector.vector_fp import VectorFPGradientFunction, MappingOnlyMatrix


def build_optimizer(lr, sigma, gamma, beta1, beta2, epsilon, patience, cost_limit_factor):
    analysis = HolisticFPAnalysis(limit_factor=10, reset=False)
    handler = MappingOnlyExtractor()
    cost = InvslackCost(parameter_handler=handler, analysis=analysis)
    stop = ThresholdStopFunction(limit=100, patience=patience)
    grad = VectorFPGradientFunction(
        scenarios_builder=MappingOnlyMatrix(),
        sigma=sigma,
        cost_limit_factor=cost_limit_factor)
    update = NoisyAdam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, gamma=gamma)
    pd = PDAssignment(normalize=True)
    optimizer = GradientDescentOptimizer(
        parameter_handler=handler,
        cost_function=cost,
        stop_function=stop,
        gradient_function=grad,
        update_function=update,
        verbose=False)

    class _Opt:
        def apply(self, system):
            pd.apply(system)
            optimizer.apply(system)
    return _Opt()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for FP mapping-only gradient descent")
    parser.add_argument("-o", "--output-dir",
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Output directory (default: script directory)")
    parser.add_argument("-t", "--threads", type=int, default=6,
                        help="Number of worker processes (default: 6)")
    parser.add_argument("--top-n", type=int, default=15,
                        help="Number of top combinations in bar chart (default: 15)")
    args = parser.parse_args()

    utilizations = np.linspace(0.5, 0.9, 20)
    utilization = utilizations[8]

    print(f"Started:  {datetime.now()}")
    print(f"CLI args: threads={args.threads}, top_n={args.top_n}, output_dir={args.output_dir}")
    print(f"Scenario: fp-mapping-only, target_utilization={utilization}")
    print(f"Modules:  MappingOnlyExtractor, InvslackCost, ThresholdStopFunction")
    print(f"          VectorFPGradientFunction(MappingOnlyMatrix), NoisyAdam")
    print(f"Setup:    PDAssignment(normalize=True), unbalance_contended, set_utilization={utilization}")
    print()

    rnd = Random(42)
    size = (3, 4, 3)
    n = 100
    systems = [get_system(size, rnd, balanced=False, name=str(i),
                          deadline_factor_min=0.5,
                          deadline_factor_max=1) for i in range(n)]
    
    for s in systems:
        set_utilization(s, utilization)
        unbalance_contended(s)

    param_grid = {
        "lr": [1.5, 3.0],
        "sigma": [1.0, 3.0],
        "gamma": [0.1, 0.3],
        "beta1": [0.9],
        "beta2": [0.999],
        "epsilon": [0.01, 0.1],
        "patience": [None],
        "cost_limit_factor": [1],
    }

    tuner = GradientHyperTuner(
        name="fp_mapping_only",
        systems=systems,
        param_grid=param_grid,
        build_optimizer=build_optimizer,
        threads=args.threads,
        output_dir=args.output_dir,
        top_n=args.top_n,
        verbose=True,
    )
    tuner.run()
    print(f"\nFinished: {datetime.now()}")
