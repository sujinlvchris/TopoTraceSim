#!/usr/bin/env bash
# Routed Switch-style MoE benchmark with real top-1 expert assignment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QINLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TORUS="2x2"
TOKENS_PER_SOURCE=128
D_MODEL=768
D_FF=3072
NUM_EXPERTS=8
EXPERT_CAPACITY=64
ROUTER_SKEW=0.5
ROUTER_SIGNAL=3.0
ROUTER_NOISE=1.0
SEED=20260609
SCHEDULER="dimrotation"
FLIT_SIZE=64
CLOCK_GHZ=1.0
SLACK=0.10
HARDWARE_CONFIG="/mnt/sdb1/wyf/TopoTraceSim/configs/noi_hbm_fixed.yaml"
POPNET_BIN="${POPNET_BIN:-/mnt/sdb1/wyf/TopoTraceSim/third_party/popnet_anytopo/build/popnet}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --torus) TORUS="$2"; shift 2 ;;
    --tokens-per-source) TOKENS_PER_SOURCE="$2"; shift 2 ;;
    --d-model) D_MODEL="$2"; shift 2 ;;
    --d-ff) D_FF="$2"; shift 2 ;;
    --num-experts) NUM_EXPERTS="$2"; shift 2 ;;
    --expert-capacity) EXPERT_CAPACITY="$2"; shift 2 ;;
    --router-skew) ROUTER_SKEW="$2"; shift 2 ;;
    --router-signal) ROUTER_SIGNAL="$2"; shift 2 ;;
    --router-noise) ROUTER_NOISE="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --scheduler) SCHEDULER="$2"; shift 2 ;;
    --hardware-config) HARDWARE_CONFIG="$2"; shift 2 ;;
    --popnet-bin) POPNET_BIN="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

ARY=${TORUS%%x*}
DIMS=$(awk -F 'x' '{print NF}' <<<"${TORUS}")
RUN_TAG="switch_base8_routed_${TORUS}_tok${TOKENS_PER_SOURCE}_skew${ROUTER_SKEW}_${SCHEDULER}"

CSV="${QINLE_ROOT}/traces/${RUN_TAG}.csv"
ROUTE_JSON="${QINLE_ROOT}/traces/${RUN_TAG}_routes.json"
BENCH_DIR="${QINLE_ROOT}/traces/${RUN_TAG}"
POPNET_STDOUT="${QINLE_ROOT}/output/${RUN_TAG}.stdout"
POPNET_LOG_DST="${QINLE_ROOT}/output/${RUN_TAG}.log"
METRICS_JSON="${QINLE_ROOT}/results/${RUN_TAG}.json"
BREAKDOWN_JSON="${QINLE_ROOT}/results/${RUN_TAG}_breakdown.json"

mkdir -p "${QINLE_ROOT}/traces" "${QINLE_ROOT}/output" "${QINLE_ROOT}/results"

echo "===== Routed Switch-style MoE benchmark ====="
echo "  torus         : ${TORUS}"
echo "  tokens/source : ${TOKENS_PER_SOURCE}"
echo "  d_model/d_ff  : ${D_MODEL}/${D_FF}"
echo "  experts/cap   : ${NUM_EXPERTS}/${EXPERT_CAPACITY}"
echo "  router skew   : ${ROUTER_SKEW}"
echo "  scheduler     : ${SCHEDULER}"

echo "===== [1/5] PyTorchSim routed MoE trace ====="
SWITCH_DRIVER=run_switch_routed_moe_layer.py \
  bash "${QINLE_ROOT}/pytorchsim/docker_switch_entry.sh" \
    --dims "${DIMS}" --ary "${ARY}" \
    --tokens-per-source "${TOKENS_PER_SOURCE}" \
    --d-model "${D_MODEL}" \
    --d-ff "${D_FF}" \
    --num-experts "${NUM_EXPERTS}" \
    --expert-capacity "${EXPERT_CAPACITY}" \
    --router-skew "${ROUTER_SKEW}" \
    --router-signal "${ROUTER_SIGNAL}" \
    --router-noise "${ROUTER_NOISE}" \
    --seed "${SEED}" \
    --flit-size "${FLIT_SIZE}" \
    --out "traces/$(basename "${CSV}")" \
    --router-out "traces/$(basename "${ROUTE_JSON}")"

echo "===== [2/5] Build bench (${SCHEDULER}) ====="
python3 "${QINLE_ROOT}/scripts/build_bench.py" \
  --csv "${CSV}" --dims "${DIMS}" --ary "${ARY}" \
  --scheduler "${SCHEDULER}" --out-dir "${BENCH_DIR}" --slack "${SLACK}"

echo "===== [3/5] PopNet ====="
RUN_CWD="${QINLE_ROOT}/output/run_${RUN_TAG}"
mkdir -p "${RUN_CWD}"
set +e
( cd "${RUN_CWD}" && \
  "${POPNET_BIN}" -A "${ARY}" -c "${DIMS}" -V 3 -B 12 -O 12 -F 4 \
    -L 1000 -T 3000000 -r 1 \
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
  --noi-wait-source a2a-window \
  --out "${BREAKDOWN_JSON}"

echo "DONE: ${BREAKDOWN_JSON}"
echo "ROUTES: ${ROUTE_JSON}"
