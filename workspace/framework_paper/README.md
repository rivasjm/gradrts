# Gradient Descent framework evaluations (paper)

This directory contains the evaluations used in the evaluation section of the paper
(`src/06_evaluation.tex`), the scripts that run them, the generated data, and the scripts
that produce the figures.

## Evaluated scenarios

Three scenarios are evaluated, each optimizing a different set of system parameters with
the GDPA (Gradient Descent Parameter Assignment) framework:

| Scenario | Directory | Optimized parameters | Analysis |
|----------|-----------|----------------------|----------|
| **FP**   | `fp/`  | Fixed priorities     | Holistic FP |
| **MAP**  | `map/` | Fixed priorities + step-to-processor mapping | Holistic FP |
| **EDF**  | `edf/` | Local deadlines      | Holistic Local EDF |

All scenarios share the same canonical pool of synthetic systems, generated in
`systems.py` with a fixed seed (42), 50 systems per size and 20 utilization levels between
50% and 90%. Sizes are predefined in `systems.py`:

| Size key | Layout (flows x tasks x processors) | Total tasks |
|----------|-------------------------------------|-------------|
| `15`     | 5 x 3 x 3                           | 15          |
| `25`     | 5 x 5 x 5                           | 25          |

Task utilizations are drawn **globally** with UUNIFAST (`utilization * n_procs` in total, so
the average per-processor utilization is the requested value), and only the
task-to-processor mapping differs between variants. The balanced and unbalanced
populations therefore share exactly the same task set (periods, deadlines and
utilizations); only the initial mapping and load distribution change:

- **balanced** (`get_systems_<size>()`): tasks are assigned round-robin by decreasing
  utilization, so every processor gets the same number of tasks. Loads are equalized at
  each utilization level by `set_utilization`.
- **unbalanced** (`get_systems_<size>_unbalanced()`): a contended mapping with uneven
  per-processor load (`unbalance_contended`), swept with `set_system_utilization` so the
  initial mapping and load distribution are kept as utilization grows.

Scenarios:

- **FP**: balanced systems.
- **MAP**: two variants: balanced (`map-<size>`) and unbalanced (`map-unbalanced-<size>`).
- **EDF**: balanced systems with all processors switched to local EDF.

## Validation scripts

Each scenario script takes an optional size argument (default `15`) and writes its data
into `<scenario>/<scenario>-<size>/`:

- `fp/fp.py [15|25]` -> `fp/fp-<size>/`
- `map/map.py [15|25] [--unbalanced]` -> `map/map-<size>/` or `map/map-unbalanced-<size>/`
- `edf/edf.py [15|25]` -> `edf/edf-<size>/`

Methods compared per scenario (column names in the `.xlsx` files):

| Scenario | Methods |
|----------|---------|
| FP (`fp-15`)    | `gdpa`, `hopa`, `pd`, `bf` |
| FP (`fp-25`)    | `gdpa`, `hopa`, `pd` |
| MAP   | `pd`, `hopa`, `gdpa-100`, `gdpa-200` |
| EDF   | `pd`, `hopa`, `gdpa` |

- `gdpa`: GDPA framework (in MAP it is evaluated with 100 and 200 maximum iterations).
- `hopa`: the iterative HOPA algorithm.
- `pd`: proportional deadlines assignment (non-iterative).
- `bf`: exhaustive priority search (FP only, size 15 only). It evaluates all
  $5!^3 = 1{,}728{,}000$ priority orderings (the `batch_size=10000` in the code is only
  the processing batch size).

## Generated data

Each scenario produces, in its output directory, one `.xlsx` file per metric:

- `*_schedulables.xlsx`: number of schedulable systems (max. 50) per utilization level and
  method (20 rows x number of methods).
- `*_times.xlsx`: global mean time per utilization level and method.
- `*_times_success.xlsx`: mean time **only over the systems for which a schedulable
  solution was found** (time-to-success) per utilization level and method. This is the one
  used in the time figures.

Diagnostic PNGs are also generated (`*_schedulables.png`, `*_schedulables_summary.png`,
`*_efficiency.png`).

## Figure scripts

The figure scripts iterate over every size in `systems.py` and produce one figure per
size, with four panels: FP, MAP, MAP unbalanced and EDF. Shared configuration lives in
`figures.py` (`SIZES`, `SCENARIOS`, `METHOD_STYLES`).

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `charts.py` | `*_schedulables.xlsx` | `schedulables_<size>.pdf` / `.png` | Number of schedulable systems per utilization |
| `times.py` | `*_times_success.xlsx` | `times_<size>.pdf` / `.png` | Mean time to schedulable solution per utilization (log y-axis) |
| `charts-efficiency.py` | `*_schedulables.xlsx` + `*_times_success.xlsx` | `efficiency_<size>.pdf` / `.png` | Aggregate total schedulability vs. total time (scatter, log x-axis) |

Only `schedulables` and `times` are currently used in the paper; the `efficiency` figure is
not included.

### Shown methods and colors convention

The figures show a subset of the methods and rename columns:

- **FP**: `gdpa`, `hopa`, `pd`, and `bf` when present.
- **MAP** (both variants): `pd`, `hopa`, `gdpa-100`, `gdpa-200`.
- **EDF**: `pd`, `hopa`, `gdpa`.

Colors are consistent across the figures (defined in `METHOD_STYLES`):

| Method | Color |
|--------|-------|
| `gdpa`, `gdpa-100` | blue `#0000FF` |
| `gdpa-200` | orange `#FF8C00` |
| `hopa` | green `#008000` |
| `pd` | brown `#8B4513` |
| `bf` | red `#FF0000` |

## How to run

Use the `code/.venv` virtualenv. From the `code` directory:

```bash
# 1. (re)generate the data of a scenario and size
python workspace/framework_paper/fp/fp.py 15
python workspace/framework_paper/map/map.py 15
python workspace/framework_paper/map/map.py 15 --unbalanced
python workspace/framework_paper/edf/edf.py 15

# 2. regenerate the figures (one per size)
python workspace/framework_paper/charts.py
python workspace/framework_paper/times.py
python workspace/framework_paper/charts-efficiency.py
```

The evaluations are expensive (EDF is the slowest scenario). To copy a figure into the
paper:

```bash
cp workspace/framework_paper/schedulables_15.pdf src/figs/  # adjust destination
cp workspace/framework_paper/times_15.pdf src/figs/
```
