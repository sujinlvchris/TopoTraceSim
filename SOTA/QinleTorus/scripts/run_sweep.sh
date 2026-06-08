#!/usr/bin/env bash
# Run the full QinleTorus sweep: msg ∈ {8KB,16KB,64KB,256KB} × torus ∈ {4x4,8x8}
# × scheduler ∈ {direct,dimrotation}.  For each (msg,torus) the chunked CSV is
# produced ONCE and reused by both schedulers.
#
# Usage:
#   bash scripts/run_sweep.sh                     # full sweep
#   MSGS="8KB 16KB" TORI="4x4" bash scripts/run_sweep.sh
#   bash scripts/run_sweep.sh --smoke             # just 4x4, 8KB, both schedulers

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QINLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MSGS_DEFAULT="8KB 16KB 64KB 256KB"
TORI_DEFAULT="4x4 8x8"
SCHEDS_DEFAULT="direct dimrotation"

SMOKE=0
for arg in "$@"; do
  case "$arg" in
    --smoke) SMOKE=1 ;;
  esac
done

MSGS="${MSGS:-${MSGS_DEFAULT}}"
TORI="${TORI:-${TORI_DEFAULT}}"
SCHEDS="${SCHEDS:-${SCHEDS_DEFAULT}}"
if [[ "${SMOKE}" -eq 1 ]]; then
  MSGS="8KB"
  TORI="4x4"
fi

echo "sweep:"
echo "  MSGS  : ${MSGS}"
echo "  TORI  : ${TORI}"
echo "  SCHEDS: ${SCHEDS}"

for torus in ${TORI}; do
  for msg in ${MSGS}; do
    csv="${QINLE_ROOT}/traces/torus_${torus}_${msg}.csv"
    first=1
    for sched in ${SCHEDS}; do
      if [[ "${first}" -eq 1 ]]; then
        bash "${SCRIPT_DIR}/run_one.sh" --torus "${torus}" --msg "${msg}" --scheduler "${sched}"
        first=0
      else
        bash "${SCRIPT_DIR}/run_one.sh" --torus "${torus}" --msg "${msg}" --scheduler "${sched}" --skip-pytorchsim
      fi
    done
  done
done

echo "===== sweep done; rendering figures ====="
python3 "${QINLE_ROOT}/analysis/plot.py" \
    --results-dir "${QINLE_ROOT}/results" \
    --out-dir    "${QINLE_ROOT}/results/figures" \
    --dims 2

echo "all figures under ${QINLE_ROOT}/results/figures"
