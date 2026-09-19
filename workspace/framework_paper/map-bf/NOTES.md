# map-bf — notes for future sessions

Context for an agent starting from scratch. `map-bf` compares the **GDPA**
framework (joint task-to-processor mapping + fixed priorities) against an
**exact brute force** on systems small enough to enumerate exhaustively.

## Population and size

`bf.py` builds the population from `SIZES` (`bf.py`), seed 42, 25 systems,
periods 100–1000, deadline factors 0.5–1, 20 utilization levels (0.5–0.9).

| size key | flows x tasks x procs | total tasks | search space `n!·C(n+p-1,p-1)` |
|---|---|---|---|
| 9  | (3,3,3) | 9  | 19,958,400 |
| 10 | (2,5,3) | 10 | 239,500,800 |
| 12 | (4,3,3) | 12 | 43,589,145,600 (~182x size 10) |

- **Unbalanced (default)**: `balanced=False` -> `unbalance_contended`
  (`examples/generator.py`), a contended mapping (staggered hi/lo targets,
  best-fit decreasing), generated at utilization 0.5. The sweep then calls
  `set_system_utilization`, which scales all WCETs; at high u some processor
  ends up > 1 and GDPA must repair it.
- **Balanced (`--balanced`)**: `balanced=True` -> `assign_balanced` (round-robin
  by decreasing utilization) and the sweep uses `set_utilization` (every
  processor set to exactly u).
- `eval_name` is `map-bf-<size>` or `map-bf-<size>-balanced`.

## Methods

`bf.py` runs, in order: `pd`, `hopa`, `gdpa-prio`, `gdpa-100`, `gdpa-200`,
`gdpa-500`, `bf`, and (`--methods bf-seq` / size 9 default) `bf-seq`.

- `gdpa-prio`: GDPA optimizing only priorities (mapping fixed).
- `gdpa-100/200/500`: **bounded multi-start** GDPA. Each restart gets `chunk`
  iterations; the number in the name is the total budget `chunk * restarts`:
  - gdpa-100 = chunk 25 x 4
  - gdpa-200 = chunk 20 x 10
  - gdpa-500 = chunk 50 x 10
  Parameters: `lr=10`, `warmup=0`, `mapping_delta=2.0`, `priority_delta=2.0`
  (large learning rate and per-block finite-difference steps, no warmup).
- `bf`: `BruteForceFPMappingAssignment` (vectorized, batch 10000, utilization
  prune). `bf-seq`: `BruteForceFPSequentialMappingAssignment` (scalar, slow).

Flags: `--size {9,10,12}`, `--balanced`, `--methods ...`, `--vector-cost`,
`--no-bf-prune`, `--batch-size`, `-u`, `--n`, `--start`, `-o`.

## Results (20 levels, 25 systems, max 500)

| method | size 9 unbal | size 10 unbal | size 9 balanced |
|---|---|---|---|
| pd | 223 | 207 | 402 |
| hopa | 265 | 257 | 431 |
| gdpa-prio | 270 | 256 | 443 |
| gdpa-100 | 425 | 360 | 458 |
| gdpa-200 | 435 | 380 | 458 |
| gdpa-500 | 453 | 393 | 464 |
| bf | 459 | 406 | 493 |
| bf-seq | 459 | — | — |
| **gap gdpa-500** | **6** | **13** | **29** |

## Key findings

- **The GDPA gap vs brute force is entirely about mapping exploration.** The
  `diagnose.py` classifier over gap cases (size 9, 5 levels) gave
  a=8 (mapping left unchanged), b=21 (converged to a worse mapping), c=0
  (priorities never the cause). gdpa stalls at a near-feasible cost early
  (~iter 30-60) and stops improving.
- **What closes the gap** (tuning study, `tune.py`, 8 gap levels, bf = 176):
  - large gradient steps: `mapping_delta`/`priority_delta` 2.0 (134 -> 158)
  - **multi-start**: restarting with different noise seeds (134 -> 169)
  - combined + `limit=500`: 172/200 (gap 4 on those levels)
  - noise is essential (Adam without noise scored 50/200)
  - warmup and gamma did not help; wider mapping margins hurt.
- **Bounded restarts** (user idea): share a fixed budget, restart every `chunk`
  iterations. `chunk=50, restarts=10, warmup=0` reaches the same quality as the
  2500-iteration version with 5x less compute.
- **Balanced vs unbalanced**: balanced systems are easier (bf 459 -> 493) but
  the **gap grows** (6 -> 29), concentrated at u >= 0.8. At u=0.9 gdpa-500
  plateaus at ~15 in both cases while bf goes 16 (unbal) -> 21 (bal): exhaustive
  search keeps finding solutions GDPA's local search misses.
- **Times**: measured on the remote. `bf` (vectorized + prune) is fast and
  gets faster at high u (prune removes most mappings), so at u >= ~0.7 bf is
  often faster than GDPA (whose multi-start runs its full budget). At low u
  GDPA is faster. So small systems are not favourable to GDPA on time; the
  argument for GDPA is scalability (bf explodes with n, GDPA does not).

## Diagnostics and things that bit us

- **Scalar `HolisticFPAnalysis` is pathologically slow** for some systems
  (fixed-point convergence when a processor's utilization is near/over 1).
  Mitigations added: `max_time` (per-call budget) and `prune_over_utilized`
  (returns a saturated infeasible cost immediately, mirroring the vectorized
  over-utilization shortcut). `prune_over_utilized` is enabled for size 10 only
  so far; enabling it changes scalar-cost results broadly.
- **HOPA** runs up to 160 scalar analyses; on some systems one analysis is
  very slow, so `hopa_mapping_fp` caps the analysis at `SCALAR_ANALYSIS_MAX_TIME`
  (0.5 s). Same for `gdpa-prio`.
- **`bf-seq`** (scalar brute force) can hang on contended systems; it uses a
  `max_time` cap. It matched the vectorized bf exactly on size 9 (459) and is
  ~5x slower on average at size 10 (useful if a slower exact baseline is needed).
- **`--vector-cost`**: using `VectorHolisticFPAnalysis` (sharing the gradient's
  cache) for the cost gives identical results and ~5-9 % faster GDPA. Optional,
  not yet the default.
- **`--no-bf-prune`** makes bf vastly slower (e.g. 14 s -> >15 min at u=0.9,
  size 10) because it enumerates over-utilized mappings too.
- **12 tasks is not viable with bf** (43.6B candidates): even balanced, an
  infeasible high-u system enumerates for tens of minutes. Size 10 is the
  practical ceiling with the vectorized brute force.

## Files

- `bf.py` — the scenario (population, methods, CLI).
- `charts.py` — schedulability figure + optimality gap (`--size`, `--balanced`).
- `times.py` — time-to-schedulable-solution figure.
- `merge_methods.py` — merge method columns from a partial `--methods` run into
  the full xlsx (source columns replace target ones; regenerates diagnostics).
- `tune.py` + `run_tuning.sh` — GDPA tuning phases (`budget`, `scale`, `push`,
  `alt`, `final`, `winner`, ...); size 9 only.
- `diagnose.py` — gap classifier (a/b/c) + GDPA cost/mapping trace.

## How to run

Remote is a Mac (`ctrpc18`) reached with `ssh despacho` (through a Lima VM that
runs FortiClient). Long runs go in `tmux` session `eval`.

```bash
# from code/, full scenario (add --balanced for the balanced variant)
python workspace/framework_paper/map-bf/bf.py --size 10
python workspace/framework_paper/map-bf/bf.py --size 9 --balanced

# only some methods into a temp dir, then merge
python workspace/framework_paper/map-bf/bf.py --size 10 --methods gdpa-500 -o /tmp/x
python workspace/framework_paper/map-bf/merge_methods.py --size 10 \
    --target workspace/framework_paper/map-bf/map-bf-10 --source /tmp/x/map-bf-10

# figures
python workspace/framework_paper/map-bf/charts.py --size 10
python workspace/framework_paper/map-bf/times.py  --size 10
```

## Open questions

- Adopt `--vector-cost` as the default for GDPA?
- Keep `bf-seq` in the scenario (only size 9)?
- How to present the times: bf is competitive/faster at small n; the scalability
  argument needs the space formula rather than measured times.
- If a bigger exact case is wanted, the next feasible step is 10 tasks; 12 needs
  a different (non-exhaustive) reference or candidate reordering.
