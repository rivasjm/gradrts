# Gradient Descent framework evaluations (paper)

This directory contains the evaluations used in the evaluation section of the paper
(`src/06_evaluation.tex`), the scripts that run them, the generated data, and the scripts
that produce the figures.

## Evaluated scenarios

Three scenarios are evaluated, each optimizing a different set of system parameters with
the GDPA (Gradient Descent Parameter Assignment) framework:

| Scenario | Directory | Optimized parameters | Analysis |
|----------|-----------|----------------------|----------|
| **FP**   | `fp/`         | Fixed priorities     | Holistic FP |
| **MAP**  | `fp-mapping/` | Fixed priorities + step-to-processor mapping | Holistic FP |
| **EDF**  | `edf-local/`  | Local deadlines      | Holistic Local EDF |

All scenarios share the same pool of synthetic systems: 5 flows x 3 tasks = 15 tasks,
3 processors, 20 utilization levels between 50% and 90%, 50 systems per level. The FP and
EDF scenarios use an initially balanced mapping (5 tasks per processor); the MAP scenario
uses an arbitrary, unbalanced initial mapping.

## Validation scripts

Each scenario has a script that generates the data:

- `fp/gradient_fp_val.py` -> generates `fp/gradient_fp_eval/`
- `fp-mapping/gradient_fp_mapping_val.py` -> generates `fp-mapping/gradient_fp_mapping_eval/`
- `edf-local/gradient_edf_local_val.py` -> generates `edf-local/gradient_edf_local_eval/`

Methods compared per scenario (column names in the `.xlsx` files):

| Scenario | Methods |
|----------|---------|
| FP    | `gdpa-vec`, `gdpa-seq`, `hopa`, `pd`, `bf` |
| MAP   | `pd`, `hopa`, `gdpa-50`, `gdpa-100`, `gdpa-200` |
| EDF   | `pd`, `hopa`, `gdpa` |

- `gdpa`: GDPA framework (in MAP it is evaluated with 50, 100 and 200 maximum iterations).
- `hopa`: the iterative HOPA algorithm.
- `pd`: proportional deadlines assignment (non-iterative).
- `bf`: exhaustive priority search (FP only). It evaluates all $5!^3 = 1{,}728{,}000$
  priority orderings (the `batch_size=10000` in the code is only the processing batch size).

## Generated data

Each scenario produces, in its `*_eval/` subdirectory, one `.xlsx` file per metric:

- `*_schedulables.xlsx`: number of schedulable systems (max. 50) per utilization level and
  method (20 rows x number of methods).
- `*_times.xlsx`: global mean time per utilization level and method.
- `*_times_success.xlsx`: mean time **only over the systems for which a schedulable
  solution was found** (time-to-success) per utilization level and method. This is the one
  used in the time figures.

Diagnostic PNGs are also generated (`*_schedulables.png`, `*_schedulables_summary.png`,
`*_efficiency.png`).

## Figure scripts

The three scripts read the `.xlsx` files and produce the figures in this directory:

| Script | Reads | Produces | Description |
|--------|-------|----------|-------------|
| `charts.py` | `*_schedulables.xlsx` | `schedulables.pdf` / `.png` | Number of schedulable systems per utilization, 3 panels (FP, MAP, EDF) |
| `times.py` | `*_times_success.xlsx` | `times.pdf` / `.png` | Mean time to schedulable solution per utilization (log y-axis), 3 panels |
| `charts-efficiency.py` | `*_schedulables.xlsx` + `*_times_success.xlsx` | `efficiency.pdf` / `.png` | Aggregate total schedulability vs. total time (scatter, log x-axis) |

Only `schedulables` and `times` are currently used in the paper; the `efficiency` figure is
not included.

### Shown methods and colors convention

The figures show a subset of the methods and rename columns:

- **FP**: `gdpa` (column `gdpa-vec`), `hopa`, `pd`, `bf`.
- **MAP**: `pd`, `hopa`, `gdpa-100`, `gdpa-200`.
- **EDF**: `pd`, `hopa`, `gdpa`.

Colors are consistent across the three figures (defined in `METHOD_STYLES` /
`METHOD_COLORS`):

| Method | Color |
|--------|-------|
| `gdpa`, `gdpa-100` | blue `#0000FF` |
| `gdpa-200` | orange `#FF8C00` |
| `hopa` | green `#008000` |
| `pd` | brown `#8B4513` |
| `bf` | red `#FF0000` |

## How to run

Use the repository venv:

```bash
# 1. (re)generate the data of a scenario
python fp/gradient_fp_val.py
python fp-mapping/gradient_fp_mapping_val.py
python edf-local/gradient_edf_local_val.py

# 2. regenerate the figures
python charts.py
python times.py
python charts-efficiency.py
```

The evaluations are expensive (EDF is the slowest scenario). To copy the figures into the
paper:

```bash
cp schedulables.pdf ../../figs/
cp times.pdf ../../figs/
```