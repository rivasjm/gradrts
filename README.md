# GradRTS

Real-time systems schedulability analysis with gradient-based optimization.

## Setup

Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Running experiments

```bash
python workspace/gradient_fp_validation/gradient_fp_val.py
python workspace/gradient_fp_mapping_validation/gradient_fp_mapping_val.py
python workspace/gradient_edf_local_validation/gradient_edf_local_val.py
python workspace/gradient_edf_local_mapping_validation/gradient_edf_local_mapping_val.py
```

## Running tests

```bash
python -m pytest tests/
```
