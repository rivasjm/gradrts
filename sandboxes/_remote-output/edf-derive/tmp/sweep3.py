import common as C
from frozen import harness
import time

def share_scale(system, g, lam):
    for f in system.flows:
        total = sum(t.wcet for t in f.tasks)
        ds, prev = [], 0.0
        for t in f.tasks:
            s = t.wcet * f.deadline / total
            ds.append(s + g * prev)
            prev += s
        scale = lam * f.deadline / sum(ds)
        for t, d in zip(f.tasks, ds):
            t.deadline = d * scale

def cum_lam(system, lam):
    # d_i = lam * prefix_wcet_i  (== lam*D*cumshare_i)
    for f in system.flows:
        pref = 0.0
        for t in f.tasks:
            pref += t.wcet
            t.deadline = lam * pref

def make(fn, *a):
    return lambda system: fn(system, *a)

cands = {}
for g in (0.0, 0.25, 0.5, 0.75, 1.0):
    for lam in (1.75, 2.0, 2.25, 2.5):
        cands[f"sh_g{g}_l{lam}"] = make(share_scale, g, lam)
for lam in (0.5, 0.75, 1.0):
    cands[f"cum_l{lam}"] = make(cum_lam, lam)

t0 = time.perf_counter()
rep = harness.evaluate(cands)
for s, n in sorted(((m["schedulable"], n) for n, m in rep["techniques"].items()), reverse=True):
    print(f"{n:18s} {s}")
print("elapsed", round(time.perf_counter()-t0, 1))
