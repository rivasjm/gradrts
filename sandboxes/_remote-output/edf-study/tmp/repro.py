"""Reproduce the AGENTS.md reference tables (γ and λ) and the reference techniques."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa

print("--- reference techniques ---")
run({"pd": pd, "pd_global": pd_global, "equal": equal,
     "period_prop": period_prop})

print("\n--- gamma family ---")
run({f"g{g:.1f}": pd_gamma(g) for g in
     [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]})

print("\n--- lambda family ---")
run({f"l{lam}": pd_lambda(lam) for lam in
     [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]})
