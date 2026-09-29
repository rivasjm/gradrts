# Gradient Descent framework evaluations (paper)

## Run all evaluations

From the `code/` directory, with the virtualenv active (`source .venv/bin/activate`):

```bash
# FP
python workspace/framework_paper/fp/fp.py 15
python workspace/framework_paper/fp/fp.py 25

# EDF
python workspace/framework_paper/edf/edf.py 15
python workspace/framework_paper/edf/edf.py 25

# MAP: exact baseline + scalability (used in the paper)
python workspace/framework_paper/map-bf-scalability/map-bf-scalability.py

# MAP: additional runs (kept for reference, not used in the paper)
python workspace/framework_paper/map/map.py 15
python workspace/framework_paper/map/map.py 25
python workspace/framework_paper/map/map.py 15 --unbalanced
python workspace/framework_paper/map/map.py 25 --unbalanced
python workspace/framework_paper/map-bf/bf.py --size 10

# Paper figures
python workspace/framework_paper/figures-fp.py
python workspace/framework_paper/figures-edf.py
python workspace/framework_paper/figures-map.py
```

Each scenario writes into `<scenario>/<scenario>-<size>/`. EDF is the slowest
scenario.

Alternatively, run the FP, MAP and EDF data-generation scenarios sequentially
with:

```bash
bash workspace/framework_paper/run_all.sh
```

`run_all.sh` also accepts an optional subset of scenarios (`fp`, `map`, `edf`);
with no arguments it runs all three, and it regenerates the figures at the end:

```bash
bash workspace/framework_paper/run_all.sh map       # only MAP (balanced + unbalanced)
bash workspace/framework_paper/run_all.sh fp edf    # selected scenarios
```

This directory contains the evaluations used in the evaluation section of the
paper (`src/06_evaluation.tex`), the scripts that run them, the generated data,
and the scripts that produce the figures.

## Evaluated scenarios

Three scenarios are evaluated, each optimizing a different set of system
parameters with the GDPA (Gradient Descent Parameter Assignment) framework:

| Scenario | Directory | Optimized parameters | Analysis |
|----------|-----------|----------------------|----------|
| **FP**   | `fp/`  | Fixed priorities     | Holistic FP |
| **EDF**  | `edf/` | Local deadlines      | Holistic Local EDF |
| **MAP**  | `map-bf-scalability/` | Task-to-processor mapping + fixed priorities | Holistic FP |

All scenarios share the same canonical pool of synthetic systems, generated in
`systems.py` with a fixed seed (42), **except MAP**, which uses its own nested
pool (see `map-bf-scalability/systems.py`) at a fixed utilization:

| Size key | Layout (flows x tasks x processors) | Total tasks |
|----------|-------------------------------------|-------------|
| `15`     | 5 x 3 x 3                           | 15          |
| `25`     | 5 x 5 x 5                           | 25          |

Task utilizations are drawn **globally** with UUNIFAST (`utilization * n_procs`
in total, so the average per-processor utilization is the requested value). The
FP and EDF populations use 50 systems per size and 20 utilization levels between
50% and 90%, with a balanced initial mapping. The balanced and unbalanced
populations share exactly the same task set (periods, deadlines and
utilizations); only the initial mapping and load distribution differ:

- **balanced** (`get_systems_<size>()`): tasks are assigned round-robin by
  decreasing utilization, so every processor gets the same number of tasks.
- **unbalanced** (`get_systems_<size>_unbalanced()`): a contended mapping with
  uneven per-processor load (`unbalance_contended`), swept with
  `set_system_utilization`.

Scenarios:

- **FP**: balanced systems.
- **EDF**: balanced systems with all processors switched to local EDF.
- **MAP**: a nested population of 25 systems at a fixed utilization (0.75),
  whose task count grows from a 4-task base up to 16, with a contended initial
  mapping that GDPA must repair. The brute force provides an exact reference at
  the sizes where it is still tractable, and its timeout beyond ~12 tasks is the
  scalability result.

## Validation scripts

Each scenario script takes an optional size argument (default `15`) and writes
its data into `<scenario>/<scenario>-<size>/`:

- `fp/fp.py [15|25]` -> `fp/fp-<size>/`
- `edf/edf.py [15|25]` -> `edf/edf-<size>/`
- `map/map.py [15|25] [--unbalanced]` -> `map/map-<size>/` or
  `map/map-unbalanced-<size>/`
- `map-bf-scalability/map-bf-scalability.py` -> `map-bf-scalability-<u>/`

Methods compared per scenario (column names in the `.xlsx` files):

| Scenario | Methods |
|----------|---------|
| FP (`fp-15`)    | `gdpa`, `hopa`, `pd`, `bf` |
| FP (`fp-25`)    | `gdpa`, `hopa`, `pd` |
| EDF   | `pd`, `hopa`, `gdpa` |
| MAP   | `pd`, `hopa`, `gdpa-prio`, `gdpa-100`, `gdpa-200`, `gdpa-500`, `bf` |

- `gdpa`: GDPA framework. In the paper it runs a single budget (100 iterations);
  in the MAP scalability pool the bounded multi-start variants `gdpa-100`,
  `gdpa-200` and `gdpa-500` share a total iteration budget restarting every
  25/20/50 iterations, and `gdpa-prio` optimizes only priorities keeping the
  mapping fixed.
- `hopa`: the iterative HOPA algorithm.
- `pd`: proportional deadlines assignment (non-iterative).
- `bf`: exhaustive search over priorities (FP size 15 only; it evaluates all
  $5!^3 = 1{,}728{,}000$ orderings) and over mappings + priorities (MAP, where
  it is only tractable at small task counts).

## Generated data

Each scenario produces, in its output directory, one `.xlsx` file per metric:

- `*_schedulables.xlsx`: number of schedulable systems (max. 50) per utilization
  level and method (20 rows x number of methods). The MAP scalability workbook
  instead stores three sheets (`schedulable`, `mean_time`, `finished`), each
  `tool x tasks`.
- `*_times.xlsx`: global mean time per utilization level and method.
- `*_times_success.xlsx`: mean time **only over the systems for which a
  schedulable solution was found** (time-to-success) per utilization level and
  method. This is the one used in the time figures.

Diagnostic PNGs are also generated (`*_schedulables.png`,
`*_schedulables_summary.png`, `*_efficiency.png`).

## Figure scripts

Each script produces one double-column figure per scenario. The FP and EDF
figures are 2x2 grids (schedulable systems on top, mean time below, one column
per system size); the MAP figure has three panels side by side:

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `figures-fp.py` | `fp-15/25_schedulables.xlsx`, `fp-15/25_times_success.xlsx` | `fp.pdf` | 2x2 grid: schedulable systems (top) and mean time (log, bottom) for 15/25 tasks (columns); shared axes per row and column |
| `figures-edf.py` | `edf-15/25_*.xlsx` | `edf.pdf` | Same 2x2 layout as FP |
| `figures-map.py` | `map-bf-scalability-0.75-n15_processed.xlsx` | `map-scalability.pdf` | Three stacked panels (schedulable / mean time (log) / finished) vs. task count; only the methods that optimize mapping and priorities (`gdpa-100/200/500`, `bf`) are shown |

Shared configuration (method colors, loading helpers) lives in
`figures_common.py`. The `map-bf/charts.py`, `map-bf/times.py` and
`map-bf-scalability/process.py` scripts produce quick diagnostic figures for
their own runs and are **not** the paper figures.

### Shown methods and colors convention

The figures show a subset of the methods and rename columns. Colors are
consistent across all figures (defined in `figures_common.py`):

| Method | Color | Marker |
|--------|-------|--------|
| `gdpa`, `gdpa-100` | blue `#0000FF` / `#4C72FF` | `o` / `^` |
| `gdpa-200` | light blue `#7F9BFF` | `v` |
| `gdpa-500` | navy `#000080` | `D` |
| `gdpa-prio` | grey `#999999` | `+` |
| `hopa` | green `#008000` | `x` |
| `pd` | brown `#8B4513` | `s` |
| `bf` | red `#FF0000` | `*` |

The GDPA variants share the blue family because they only differ in the
iteration budget; `gdpa-prio` is grey because it keeps the mapping fixed and is
not directly comparable with the mapping variants.

## How to run

Use the `code/.venv` virtualenv. From the `code` directory:

```bash
# 1. (re)generate the data of a scenario and size
python workspace/framework_paper/fp/fp.py 15
python workspace/framework_paper/edf/edf.py 15
python workspace/framework_paper/map-bf-scalability/map-bf-scalability.py

# 2. regenerate the paper figures
python workspace/framework_paper/figures-fp.py
python workspace/framework_paper/figures-edf.py
python workspace/framework_paper/figures-map.py
```

The evaluations are expensive (EDF and the MAP scalability run are the slowest).
To copy a figure into the paper:

```bash
cp workspace/framework_paper/fp.pdf ../figs/  # adjust destination
```
