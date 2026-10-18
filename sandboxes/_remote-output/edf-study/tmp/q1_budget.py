"""Q1: budget-feasible families (sum(d_i) == D_flow)."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

print("--- budget-feasible: normalized cumulative shape (gamnorm) ---")
run({f"gn{g:.1f}": pd_gamma_norm(g) for g in
     [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]})

print("\n--- budget-feasible: back-loading weights w_k ~ beta^k ---")
run({f"bl{beta}": back_load(beta) for beta in
     [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]})

print("\n--- overspending references for contrast ---")
run({"pd": pd, "gamma0.5": pd_gamma(0.5), "lambda2": pd_lambda(2.0),
     "pd_global": pd_global})
