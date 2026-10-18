"""Fit the dumps of ``dump_deadlines.py`` to the gamma/lambda family.

For every flow, with ``s_k = C_k D_f / sum_j C_j`` and cumulative share
``cum_k = sum_{j<k} s_j``, the paper's two-lever family is

    d_k ~ lambda * s_k + (lambda * gamma) * cum_k,

so a least-squares fit of ``(s, cum)`` on ``d`` recovers ``lambda`` (scale) and
``gamma`` (partial globalisation). Prints a table per method.

Usage: .venv/bin/python workspace/edf_wad/deadline_shape.py wad-15 ...
"""
import os
import sys
import csv
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def fit_flows(path):
    flows = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["u"], row["sys"], row["flow"])
            flows[key].append((float(row["s"]), float(row["cum_s"]),
                               float(row["d"]), float(row["D"]),
                               float(row["C"])))
    gamma, lam, r2, spend, dc = [], [], [], [], []
    for key, rows in flows.items():
        rows.sort()  # by cumulative share
        X = np.array([[r[0], r[1]] for r in rows])
        y = np.array([r[2] for r in rows])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        a, b = coef
        pred = X @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2.append(1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))
        if a > 1e-9:
            lam.append(a)
            gamma.append(b / a)
        D = rows[0][3]
        spend.append(sum(r[2] for r in rows) / D)
        with np.errstate(divide="ignore"):
            dc.append(min(r[2] / r[4] for r in rows if r[4] > 0))
    return gamma, lam, r2, spend, dc


def med(x):
    x = [v for v in x if v == v]
    return float(np.median(x)) if x else float("nan")


def iqr(x):
    x = [v for v in x if v == v]
    return float(np.percentile(x, 75) - np.percentile(x, 25)) if x else float("nan")


if __name__ == "__main__":
    print(f"{'method':>14} {'gamma':>8} {'(iqr)':>7} {'lambda':>8} {'(iqr)':>7} "
          f"{'R2':>6} {'sum d/D':>8} {'min d/C':>8}")
    for name in sys.argv[1:]:
        path = name if os.path.isfile(name) else os.path.join(HERE, name, "deadlines.csv")
        gamma, lam, r2, spend, dc = fit_flows(path)
        label = os.path.basename(os.path.dirname(path))
        print(f"{label:>14} {med(gamma):8.3f} {iqr(gamma):7.3f} {med(lam):8.3f} "
              f"{iqr(lam):7.3f} {med(r2):6.3f} {med(spend):8.2f} {min(dc):8.2f}")
