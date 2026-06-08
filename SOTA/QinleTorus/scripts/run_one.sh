#!/usr/bin/env bash
# End-to-end one configuration:
#   1. PyTorchSim chunked A2A in docker (cached by --csv basename if --skip-pytorchsim)
#   2. build bench via scripts/build_bench.py
#   3. run popnet -R 1 with TXY routing
#   4. parse results into a JSON
#
# Usage:
#   bash scripts/run_one.sh --torus 4x4 --msg 16KB --scheduler dimrotation
#   bash scripts/run_one.sh --torus 4x4 --msg 16KB --scheduler direct --skip-pytorchsim

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QINLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TORUS="4x4"
MSG="16KB"
SCHEDULER="dimrotation"
SKIP_PYTORCHSIM=0
SKIP_POPNET=0
DRY_RUN=0
SLACK=0.10
FLIT_SIZE=64
CLOCK_GHZ=1.0

POPNET_BIN_DEFAULT="/mnt/sdb1/wyf/TopoTraceSim/third_party/popnet_anytopo/build/popnet"
for cand in \
    "${POPNET_BIN_DEFAULT}" \
    "${QINLE_ROOT}/../../TopoTraceSim/third_party/popnet_anytopo/build/popnet" \
    "${QINLE_ROOT}/third_party/popnet_anytopo/build/popnet"; do
  if [[ -x "${cand}" ]]; then POPNET_BIN_DEFAULT="${cand}"; break; fi
done
POPNET_BIN="${POPNET_BIN:-${POPNET_BIN_DEFAULT}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torus)            TORUS="$2"; shift 2 ;;
    --msg)              MSG="$2"; shift 2 ;;
    --scheduler)        SCHEDULER="$2"; shift 2 ;;
    --skip-pytorchsim)  SKIP_PYTORCHSIM=1; shift ;;
    --skip-popnet)      SKIP_POPNET=1; shift ;;
    --dry-run)          DRY_RUN=1; shift ;;
    --slack)            SLACK="$2"; shift 2 ;;
    --flit-size)        FLIT_SIZE="$2"; shift 2 ;;
    --clock-ghz)        CLOCK_GHZ="$2"; shift 2 ;;
    --popnet-bin)       POPNET_BIN="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# parse torus, e.g. "4x4" -> ARY=4 DIMS=2
ARY=${TORUS%%x*}
DIMS=$(awk -F 'x' '{print NF}' <<<"${TORUS}")
NODES=$(python3 -c "print(${ARY}**${DIMS})")

CONFIG_NAME="torus_${TORUS}"
CSV="${QINLE_ROOT}/traces/${CONFIG_NAME}_${MSG}.csv"
RUN_TAG="${CONFIG_NAME}_${MSG}_${SCHEDULER}"
BENCH_DIR="${QINLE_ROOT}/traces/${RUN_TAG}"
POPNET_STDOUT="${QINLE_ROOT}/output/${RUN_TAG}.stdout"
POPNET_LOG_DST="${QINLE_ROOT}/output/${RUN_TAG}.log"
METRICS_JSON="${QINLE_ROOT}/results/${RUN_TAG}.json"

mkdir -p "${QINLE_ROOT}/traces" "${QINLE_ROOT}/output" "${QINLE_ROOT}/results"

echo "===== QinleTorus run ====="
echo "  torus      : ${TORUS}  (ary=${ARY} dims=${DIMS} nodes=${NODES})"
echo "  msg_size   : ${MSG}"
echo "  scheduler  : ${SCHEDULER}"
echo "  popnet bin : ${POPNET_BIN}"
echo "  csv        : ${CSV}"
echo "  bench dir  : ${BENCH_DIR}"

if [[ "${DRY_RUN}" -eq 1 ]]; then exit 0; fi

# 1. PyTorchSim (skippable, cached by CSV existence)
if [[ "${SKIP_PYTORCHSIM}" -eq 0 && ! -f "${CSV}" ]]; then
  echo "===== [1/4] PyTorchSim chunked A2A ====="
  bash "${QINLE_ROOT}/pytorchsim/docker_entry.sh" \
       --dims "${DIMS}" --ary "${ARY}" --msg-size "${MSG}" \
       --inject-gap 0 --per-chunk-gap 0 \
       --out "traces/${CONFIG_NAME}_${MSG}.csv"
elif [[ ! -f "${CSV}" ]]; then
  echo "ERROR: CSV not found and PyTorchSim skipped: ${CSV}" >&2; exit 1
else
  echo "===== [1/4] PyTorchSim CSV cached: ${CSV} ====="
fi

# 2. Build bench
echo "===== [2/4] Build bench (${SCHEDULER}) ====="
python3 "${QINLE_ROOT}/scripts/build_bench.py" \
    --csv "${CSV}" --dims "${DIMS}" --ary "${ARY}" \
    --scheduler "${SCHEDULER}" --out-dir "${BENCH_DIR}" --slack "${SLACK}"

# 3. PopNet
if [[ "${SKIP_POPNET}" -eq 1 ]]; then
  echo "===== [3/4] PopNet (skipped) ====="
else
  echo "===== [3/4] PopNet ====="
  if [[ ! -x "${POPNET_BIN}" ]]; then
    echo "ERROR: popnet binary not executable at ${POPNET_BIN}" >&2
    exit 1
  fi
  SIM_T=$(python3 -c "import yaml,sys;d=yaml.safe_load(open('${QINLE_ROOT}/configs/${CONFIG_NAME}.yaml'));print(d['popnet']['T'])" 2>/dev/null \
            || echo 2000000)
  # popnet writes popnet.log in CWD; run in a per-tag dir to keep them separate
  RUN_CWD="${QINLE_ROOT}/output/run_${RUN_TAG}"
  mkdir -p "${RUN_CWD}"
  set +e
  ( cd "${RUN_CWD}" && \
    "${POPNET_BIN}" -A "${ARY}" -c "${DIMS}" -V 3 -B 12 -O 12 -F 4 \
        -L 1000 -T "${SIM_T}" -r 1 \
        -I "${BENCH_DIR}/bench" -R 1 \
        > "${POPNET_STDOUT}" 2>&1 )
  POPNET_RC=$?
  set -e
  if [[ -f "${RUN_CWD}/popnet.log" ]]; then
    cp -f "${RUN_CWD}/popnet.log" "${POPNET_LOG_DST}"
  fi
  if [[ "${POPNET_RC}" -ne 0 ]]; then
    echo "WARN: popnet returned ${POPNET_RC}; metrics may be partial"
  fi
fi

# 4. Metrics
echo "===== [4/4] Metrics ====="
python3 "${QINLE_ROOT}/analysis/compute_metrics.py" \
    --bench "${BENCH_DIR}/bench" \
    --csv "${CSV}" \
    --stdout "${POPNET_STDOUT}" \
    --log "${POPNET_LOG_DST}" \
    --dims "${DIMS}" --ary "${ARY}" \
    --label "${SCHEDULER}" \
    --flit-size "${FLIT_SIZE}" \
    --clock-ghz "${CLOCK_GHZ}" \
    --out "${METRICS_JSON}"

echo "DONE: ${METRICS_JSON}"
