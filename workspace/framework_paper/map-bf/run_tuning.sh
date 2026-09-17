#!/usr/bin/env bash
# Launch the map-bf GDPA tuning study and log everything for sharing.
#
# Usage (from anywhere):
#   bash workspace/framework_paper/map-bf/run_tuning.sh            # study phase
#   bash workspace/framework_paper/map-bf/run_tuning.sh baseline
#
# Environment overrides:
#   N=25        number of systems
#   THREADS=6   worker processes
#   LOG=...     log file (default: next to this script)
#   U="0.71 ..." utilization levels (default: the gap levels in tune.py)
#
# Recommended inside tmux:
#   tmux new -s map-tune
#   bash workspace/framework_paper/map-bf/run_tuning.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../../.." && pwd)"

if [ -x "$CODE/.venv/bin/python" ]; then
    PY="$CODE/.venv/bin/python"
else
    PY="python3"
fi

PHASE="${1:-study}"
N="${N:-25}"
THREADS="${THREADS:-6}"
LOG="${LOG:-$HERE/map_tuning.log}"

U_ARGS=()
if [ -n "${U:-}" ]; then
    # shellcheck disable=SC2206
    U_ARGS=(-u $U)
fi

export PYTHONUNBUFFERED=1
cd "$CODE"

echo "Interpreter: $PY"
echo "Phase: $PHASE | n=$N | threads=$THREADS | U=${U:-<default gap levels>}"
echo "Log: $LOG"
echo

"$PY" workspace/framework_paper/map-bf/tune.py \
    --phase "$PHASE" --n "$N" --threads "$THREADS" ${U_ARGS[@]+"${U_ARGS[@]}"} \
    2>&1 | tee -a "$LOG"

echo
echo "Done. Share $LOG"
