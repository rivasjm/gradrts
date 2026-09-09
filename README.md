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

The paper evaluation lives in `workspace/framework_paper/`. See its
[README](workspace/framework_paper/README.md) for details. To run a scenario:

```bash
python workspace/framework_paper/fp/fp.py
python workspace/framework_paper/map/map.py
python workspace/framework_paper/edf/edf.py
```

## Running tests

```bash
python -m pytest tests/
```
