"""Tuning study for GDPA mapping exploration in map-bf-scalability.

The gap vs the brute force in this scenario is entirely a mapping-exploration
problem (GDPA never leaves the initial mapping). This script sweeps the knobs
that could change that, on a fixed-utilization pool, and reports how many
systems each configuration makes schedulable per size. ``bf`` is included as
the reference.

Configurations differ in how each restart is initialised (``init``: ``None``
keeps the pool mapping, ``random`` places tasks uniformly at random, ``mix``
alternates) and in the GDPA steps (``mapping_delta``, ``priority_delta``,
``lr``) and budget (``chunk`` x ``restarts``).

    .venv/bin/python workspace/framework_paper/map-bf-scalability/tune.py \
        --sizes 7 8 9 10 --threads 8
"""

import argparse
import json
import sys
import time
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path
from random import Random

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "map-bf"))  # real map-bf tools

import bf  # noqa: E402
import systems  # noqa: E402
from examples.generator import unbalance_contended  # noqa: E402

STEPS = dict(lr=10.0, warmup=0.0, mapping_delta=2.0, priority_delta=2.0,
             prune_over_utilized=True, vector_cost=True)
CONFIGS = {
    "gdpa-500":    dict(init=None, chunk=50, restarts=10, **STEPS),
    "md-none":     dict(init=None, chunk=50, restarts=10, **{**STEPS, "mapping_delta": None}),
    "md5":         dict(init=None, chunk=50, restarts=10, **{**STEPS, "mapping_delta": 5.0}),
    "md10":        dict(init=None, chunk=50, restarts=10, **{**STEPS, "mapping_delta": 10.0}),
    "rnd-md2":     dict(init="random", chunk=50, restarts=10, **{**STEPS, "mapping_delta": 2.0}),
    "rnd-md5":     dict(init="random", chunk=50, restarts=10, **{**STEPS, "mapping_delta": 5.0}),
    "rnd-md10":    dict(init="random", chunk=50, restarts=10, **{**STEPS, "mapping_delta": 10.0}),
    "rnd-md5-r20": dict(init="random", chunk=50, restarts=20, **{**STEPS, "mapping_delta": 5.0}),
    "mix-md5":     dict(init="mix", chunk=50, restarts=10, **{**STEPS, "mapping_delta": 5.0}),
    "bf":          None,
}
ORDER = list(CONFIGS)


def run_ms(system, init, chunk, restarts, **steps):
    for r in range(restarts):
        candidate = deepcopy(system)
        mode = init
        if init == "mix":
            mode = "random" if r % 2 == 0 else "unbalance"
        if mode == "random":
            rnd = Random(100 + r)
            for task in candidate.tasks:
                task.processor = candidate.processors[rnd.randrange(len(candidate.processors))]
        elif mode == "unbalance":
            unbalance_contended(candidate)
        if bf._gdpa_mapping(candidate, limit=chunk, seed=1 + r, **steps):
            return True
    return False


def _run(task):
    name, size, system = task
    if name == "bf":
        return name, size, bool(bf.bf_mapping(deepcopy(system), batch_size=10000, prune=True))
    cfg = dict(CONFIGS[name])
    init = cfg.pop("init")
    chunk = cfg.pop("chunk")
    restarts = cfg.pop("restarts")
    return name, size, bool(run_ms(system, init, chunk, restarts, **cfg))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[7, 8, 9, 10])
    parser.add_argument("--systems", type=int, default=systems.N_SYSTEMS)
    parser.add_argument("--utilization", type=float, default=systems.UTILIZATION)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--configs", nargs="+", default=ORDER)
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    pool = systems.generate_pool(n_systems=args.systems,
                                 utilization=args.utilization, verbose=True)
    index = {size: systems.SIZES.index(size) for size in args.sizes}
    tasks = [(name, size, pool[i][index[size]])
             for name in args.configs for size in args.sizes
             for i in range(args.systems)]

    solved = {name: {size: 0 for size in args.sizes} for name in args.configs}
    start = time.time()
    done = 0
    with Pool(args.threads) as mp:
        for name, size, ok in mp.imap_unordered(_run, tasks):
            solved[name][size] += ok
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(tasks)} done ({time.time()-start:.0f}s)", flush=True)

    header = "config".ljust(14) + "".join(f"{size:>6}" for size in args.sizes)
    print("\n" + header)
    for name in args.configs:
        row = name.ljust(14) + "".join(f"{solved[name][size]:>6}" for size in args.sizes)
        print(row)

    output = Path(args.output) if args.output else HERE / "tuning" / f"tune-{args.utilization:g}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"sizes": args.sizes, "solved": solved}, indent=2))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    main()
