#!/usr/bin/env bash
# Run every framework_paper evaluation scenario sequentially, then regenerate
# the figures. Results are written next to each scenario script.
#
# Usage (from this directory, code/workspace/framework_paper):
#   ./run_all.sh
# It can also be invoked from anywhere; from code/:
#   bash workspace/framework_paper/run_all.sh
#
# To see the output live and also save it to a log (recommended for long runs):
#   ./run_all.sh 2>&1 | tee run_all.log
# Inside tmux: start a session, run the above, detach with Ctrl-b d and
# reattach with `tmux attach -t <session>`. Use `tee -a run_all.log` to append
# instead of overwriting.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../.." && pwd)"

if [ -x "$CODE/.venv/bin/python" ]; then
    PY="$CODE/.venv/bin/python"
else
    PY="python3"
fi

echo "Interpreter: $PY"
export PYTHONUNBUFFERED=1
cd "$CODE"

for size in 15 25; do
    echo "=== FP size $size ==="
    "$PY" workspace/framework_paper/fp/fp.py "$size"
done

for size in 15 25; do
    echo "=== MAP size $size ==="
    "$PY" workspace/framework_paper/map/map.py "$size"
    echo "=== MAP unbalanced size $size ==="
    "$PY" workspace/framework_paper/map/map.py "$size" --unbalanced
done

for size in 15 25; do
    echo "=== EDF size $size ==="
    "$PY" workspace/framework_paper/edf/edf.py "$size"
done

echo "=== Figures ==="
"$PY" workspace/framework_paper/charts.py
"$PY" workspace/framework_paper/times.py
"$PY" workspace/framework_paper/charts-efficiency.py

echo "All evaluations and figures finished."
