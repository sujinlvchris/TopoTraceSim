#!/usr/bin/env bash
# Routed MoE-layer experiment:
#   1. run MoE expert projection chunks through PyTorchSim;
#   2. schedule the resulting chunked A2A trace with DimRotation;
#   3. replay in PopNet;
#   4. compute A2A end-to-end breakdown.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QINLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TORUS="2x2"
TOKENS_PER_REMOTE_EXPERT=64
HIDDEN_SIZE=64
SCHEDULER="dimrotation"
FLIT_SIZE=64
CLOCK_GHZ=1.0
SLACK=0.10
HARDWARE_CONFIG="/mnt/sdb1/wyf/TopoTraceSim/configs/noi_hbm_reconfigurable.yaml"

POPNET_BIN_DEFAULT="/mnt/sdb1/wyf/TopoTraceSim/third_party/popnet_anytopo/build/popnet"
POPNET_BIN="${POPNET_BIN:-${POPNET_BIN_DEFAULT}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torus) TORUS="$2"; shift 2 ;;
    --tokens-per-remote-expert) TOKENS_PER_REMOTE_EXPERT="$2"; shift 2 ;;
    --hidden-size) HIDDEN_SIZE="$2"; shift 2 ;;
    --scheduler) SCHEDULER="$2"; shift 2 ;;
    --flit-size) FLIT_SIZE="$2"; shift 2 ;;
    --clock-ghz) CLOCK_GHZ="$2"; shift 2 ;;
    --slack) SLACK="$2"; shift 2 ;;
    --hardware-config) HARDWARE_CONFIG="$2"; shift 2 ;;
    --popnet-bin) POPNET_BIN="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

ARY=${TORUS%%x*}
DIMS=$(awk -F 'x' '{print NF}' <<<"${TORUS}")
NODES=$(python3 -c "print(${ARY}**${DIMS})")

CONFIG_NAME="torus_${TORUS}"
RUN_TAG="${CONFIG_NAME}_moe_t${TOKENS_PER_REMOTE_EXPERT}_h${HIDDEN_SIZE}_${SCHEDULER}"
CSV="${QINLE_ROOT}/traces/${CONFIG_NAME}_moe_t${TOKENS_PER_REMOTE_EXPERT}_h${HIDDEN_SIZE}.csv"
BENCH_DIR="${QINLE_ROOT}/traces/${RUN_TAG}"
POPNET_STDOUT="${QINLE_ROOT}/output/${RUN_TAG}.stdout"
POPNET_LOG_DST="${QINLE_ROOT}/output/${RUN_TAG}.log"
METRICS_JSON="${QINLE_ROOT}/results/${RUN_TAG}.json"
BREAKDOWN_JSON="${QINLE_ROOT}/results/${RUN_TAG}_breakdown.json"

mkdir -p "${QINLE_ROOT}/traces" "${QINLE_ROOT}/output" "${QINLE_ROOT}/results"

echo "===== MoE A2A breakdown run ====="
echo "  torus       : ${TORUS} (ary=${ARY} dims=${DIMS} nodes=${NODES})"
echo "  tokens/expt : ${TOKENS_PER_REMOTE_EXPERT}"
echo "  hidden size : ${HIDDEN_SIZE}"
echo "  scheduler   : ${SCHEDULER}"
echo "  hardware    : ${HARDWARE_CONFIG}"
echo "  csv         : ${CSV}"
echo "  bench dir   : ${BENCH_DIR}"

echo "===== [1/5] PyTorchSim MoE layer ====="
bash "${QINLE_ROOT}/pytorchsim/docker_moe_entry.sh" \
  --dims "${DIMS}" --ary "${ARY}" \
  --tokens-per-remote-expert "${TOKENS_PER_REMOTE_EXPERT}" \
  --hidden-size "${HIDDEN_SIZE}" \
  --flit-size "${FLIT_SIZE}" \
  --out "traces/$(basename "${CSV}")"

echo "===== [2/5] Build bench (${SCHEDULER}) ====="
python3 "${QINLE_ROOT}/scripts/build_bench.py" \
  --csv "${CSV}" --dims "${DIMS}" --ary "${ARY}" \
  --scheduler "${SCHEDULER}" --out-dir "${BENCH_DIR}" --slack "${SLACK}"

echo "===== [3/5] PopNet ====="
if [[ ! -x "${POPNET_BIN}" ]]; then
  echo "ERROR: popnet binary not executable at ${POPNET_BIN}" >&2
  exit 1
fi
RUN_CWD="${QINLE_ROOT}/output/run_${RUN_TAG}"
mkdir -p "${RUN_CWD}"
set +e
( cd "${RUN_CWD}" && \
  "${POPNET_BIN}" -A "${ARY}" -c "${DIMS}" -V 3 -B 12 -O 12 -F 4 \
    -L 1000 -T 2000000 -r 1 \
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

echo "===== [4/5] Metrics ====="
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

echo "===== [5/5] Breakdown ====="
python3 "${QINLE_ROOT}/analysis/compute_moe_breakdown.py" \
  --csv "${CSV}" \
  --metrics "${METRICS_JSON}" \
  --hardware-config "${HARDWARE_CONFIG}" \
  --clock-ghz "${CLOCK_GHZ}" \
  --out "${BREAKDOWN_JSON}"

echo "DONE: ${BREAKDOWN_JSON}"
