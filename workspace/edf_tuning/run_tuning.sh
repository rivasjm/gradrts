#!/usr/bin/env bash
# Launch the EDF GDPA tuning study (size 25) and log everything for sharing.
#
# Usage (from anywhere; paths are resolved relative to this script):
#   bash workspace/edf_tuning/run_tuning.sh                  # handlers + update
#   bash workspace/edf_tuning/run_tuning.sh handlers         # one phase
#   bash workspace/edf_tuning/run_tuning.sh handlers update gradient stop
#
# Environment overrides:
#   U="0.66 0.68 0.70"   utilization levels   (default)
#   N=50                 number of systems    (default)
#   SIZE=25              system size          (default)
#   THREADS=6            worker processes     (default)
#   LOG=edf_tuning.log   log file, appended   (default: next to this script)
#
# Recommended: run inside tmux (log is unbuffered and also shown on screen):
#   tmux new -s edf-tune
#   bash workspace/edf_tuning/run_tuning.sh
#   # Ctrl-b d to detach, `tmux attach -t edf-tune` to come back
#
# High utilizations (>= 0.72) are intentionally avoided: they are much slower
# and the interesting differences already show at low/medium utilization.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$(cd "$HERE/../.." && pwd)"

if [ -x "$CODE/.venv/bin/python" ]; then
    PY="$CODE/.venv/bin/python"
else
    PY="python3"
fi

PHASES=("$@")
if [ ${#PHASES[@]} -eq 0 ]; then
    PHASES=(handlers update)
fi

U="${U:-0.66 0.68 0.70}"
N="${N:-50}"
SIZE="${SIZE:-25}"
THREADS="${THREADS:-6}"
LOG="${LOG:-$HERE/edf_tuning.log}"

export PYTHONUNBUFFERED=1
cd "$CODE"

echo "Interpreter: $PY"
echo "Phases: ${PHASES[*]} | u=$U | n=$N | size=$SIZE | threads=$THREADS"
echo "Log: $LOG"
echo

for phase in "${PHASES[@]}"; do
    echo "=== phase $phase ==="
    "$PY" workspace/edf_tuning/edf_tune.py \
        --phase "$phase" -u $U --n "$N" --size "$SIZE" --threads "$THREADS" \
        2>&1 | tee -a "$LOG"
    echo
done

echo "Done. Share $LOG"
