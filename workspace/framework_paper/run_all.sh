#!/usr/bin/env bash
# Run framework_paper evaluation scenarios sequentially, then regenerate the
# figures. Results are written next to each scenario script.
#
# Usage (from this directory, code/workspace/framework_paper):
#   ./run_all.sh              # all scenarios (fp, map, edf)
#   ./run_all.sh map          # only MAP (balanced + unbalanced)
#   ./run_all.sh fp edf       # any subset of: fp, map, edf
# It can also be invoked from anywhere; from code/:
#   bash workspace/framework_paper/run_all.sh [scenarios...]
#
# Passing all three scenarios is equivalent to passing none. Figures are only
# regenerated when every scenario runs, since they combine all of them.
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

SCENARIOS=("$@")
if [ ${#SCENARIOS[@]} -eq 0 ]; then
    SCENARIOS=(fp map edf)
fi

run_fp=false
run_map=false
run_edf=false
for scenario in "${SCENARIOS[@]}"; do
    case "$scenario" in
        fp)  run_fp=true ;;
        map) run_map=true ;;
        edf) run_edf=true ;;
        *)
            echo "Unknown scenario: $scenario (use any of: fp, map, edf)" >&2
            exit 1
            ;;
    esac
done

echo "Interpreter: $PY"
export PYTHONUNBUFFERED=1
cd "$CODE"

if $run_fp; then
    for size in 15 25; do
        echo "=== FP size $size ==="
        "$PY" workspace/framework_paper/fp/fp.py "$size"
    done
fi

if $run_map; then
    for size in 15 25; do
        echo "=== MAP size $size ==="
        "$PY" workspace/framework_paper/map/map.py "$size"
        echo "=== MAP unbalanced size $size ==="
        "$PY" workspace/framework_paper/map/map.py "$size" --unbalanced
    done
fi

if $run_edf; then
    for size in 15 25; do
        echo "=== EDF size $size ==="
        "$PY" workspace/framework_paper/edf/edf.py "$size"
    done
fi

if $run_fp && $run_map && $run_edf; then
    echo "=== Figures ==="
    "$PY" workspace/framework_paper/charts.py
    "$PY" workspace/framework_paper/times.py
    "$PY" workspace/framework_paper/charts-efficiency.py
fi

echo "Done."
