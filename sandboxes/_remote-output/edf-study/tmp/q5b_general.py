"""Q5b: sensitivity to end-to-end deadline factor and to the mapping."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from q5_general import gen_pool, make, curve  # noqa

levels = [round(0.5 + 0.4 * i / 9, 3) for i in range(10)]

print("*** deadline factor (5x3x3, balanced mapping) ***")
for (dmin, dmax) in [(0.3, 0.5), (0.5, 1.0), (0.8, 1.0), (1.0, 2.0)]:
    systems = gen_pool(5, 3, 3, population=25, dmin=dmin, dmax=dmax)
    gc, total, _ = curve({f"g{g}": make(g, 1, norm=False) for g in
                          [0.0, 0.2, 0.4, 0.6]}, systems, levels)
    lc, _, _ = curve({f"l{lam}": make(0, lam) for lam in
                      [1.0, 1.5, 2.0, 3.0]}, systems, levels)
    print(f"  dmin={dmin} dmax={dmax} total={total}")
    print(f"    gamma : {gc}")
    print(f"    lambda: {lc}")

print("\n*** mapping: balanced vs random (5x3x3, dmin/dmax default) ***")
for shuffle in [False, True]:
    systems = gen_pool(5, 3, 3, population=25, shuffle=shuffle)
    gc, total, _ = curve({f"g{g}": make(g, 1, norm=False) for g in
                          [0.0, 0.2, 0.4, 0.6]}, systems, levels)
    lc, _, _ = curve({f"l{lam}": make(0, lam) for lam in
                      [1.0, 1.5, 2.0, 3.0]}, systems, levels)
    print(f"  shuffle={shuffle} total={total}")
    print(f"    gamma : {gc}")
    print(f"    lambda: {lc}")

print("\n*** (3,5,3) lambda extended, to check the peak ***")
systems = gen_pool(3, 5, 3, population=25)
lc, total, _ = curve({f"l{lam}": make(0, lam) for lam in
                      [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]}, systems, levels)
print(f"  total={total}  {lc}")
