#!/usr/bin/env bash
# TopoTraceSim end-to-end: PyTorchSim -> CSV -> PopNet
#
#   cd /path/to/TopoTraceSim
#   bash scripts/run_a2a_full_pipeline.sh
#
#   bash scripts/run_a2a_full_pipeline.sh --convert-only   # skip PyTorchSim
#   bash scripts/run_a2a_full_pipeline.sh --smoke

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPOTRACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TORCHSIM_DIR="${TOPOTRACE_ROOT}/PyTorchSim"
POPNET_DIR="${POPNET_DIR:-${TOPOTRACE_ROOT}/third_party/popnet_anytopo}"

DOCKER_IMAGE="${TORCHSIM_DOCKER_IMAGE:-ghcr.io/psal-postech/torchsim-ci:v1.0.0}"
MSG_SIZE=16KB
FLIT_SIZE=64
INJECT_GAP=0
SIM_CYCLES=100000
HARDWARE_CONFIG="${TOPOTRACE_ROOT}/configs/noi_hbm_reconfigurable.yaml"
POPNET_BIN="${POPNET_DIR}/build/popnet"

SMOKE=0
SKIP_PYTORCHSIM=0
SKIP_POPNET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --skip-pytorchsim) SKIP_PYTORCHSIM=1; shift ;;
    --skip-popnet) SKIP_POPNET=1; shift ;;
    --convert-only) SKIP_PYTORCHSIM=1; shift ;;
    --hardware-config) HARDWARE_CONFIG="$2"; shift 2 ;;
    --msg-size) MSG_SIZE="$2"; shift 2 ;;
    --flit-size) FLIT_SIZE="$2"; shift 2 ;;
    --inject-gap) INJECT_GAP="$2"; shift 2 ;;
    --sim-cycles) SIM_CYCLES="$2"; shift 2 ;;
    --popnet-dir) POPNET_DIR="$2"; POPNET_BIN="${POPNET_DIR}/build/popnet"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ "${HARDWARE_CONFIG}" != /* ]]; then
  HARDWARE_CONFIG="${TOPOTRACE_ROOT}/${HARDWARE_CONFIG}"
fi

eval "$(
  python3 "${SCRIPT_DIR}/topotrace_hardware_config.py" \
    --config "${HARDWARE_CONFIG}" \
    --format env
)"

NODES="${TOPOTRACE_COMPUTE_NODES}"
CSV_BASENAME="a2a_n${NODES}_${MSG_SIZE}_${TOPOTRACE_NOI_INTERCONNECT_TYPE}_pytorchsim.csv"
CSV_TRACE="${TOPOTRACE_ROOT}/traces/${CSV_BASENAME}"
POPNET_TRACE_DIR="${TOPOTRACE_ROOT}/popnet_exp/traces/a2a_n${NODES}_${TOPOTRACE_NOI_INTERCONNECT_TYPE}"
POPNET_LOG="${TOPOTRACE_ROOT}/popnet_exp/logs/a2a_n${NODES}_${TOPOTRACE_NOI_INTERCONNECT_TYPE}_run.log"
HARDWARE_METADATA_JSON="${TOPOTRACE_ROOT}/traces/noi_hbm_hardware_metadata.json"
HARDWARE_SUMMARY="${TOPOTRACE_ROOT}/traces/noi_hbm_hardware_summary.txt"

EXPECTED_EVENTS=$((NODES * (NODES - 1)))
[[ "${SMOKE}" -eq 1 ]] && EXPECTED_EVENTS=1

mkdir -p "${TOPOTRACE_ROOT}/traces" "${POPNET_TRACE_DIR}" "${TOPOTRACE_ROOT}/popnet_exp/logs"

echo "TopoTraceSim root: ${TOPOTRACE_ROOT}"
echo "Hardware config: ${HARDWARE_CONFIG}"
python3 "${SCRIPT_DIR}/topotrace_hardware_config.py" \
  --config "${HARDWARE_CONFIG}" \
  --metadata-out "${HARDWARE_METADATA_JSON}" \
  --format summary | tee "${HARDWARE_SUMMARY}"

PYTORCHSIM_ARGS=(
  --nodes "${NODES}"
  --msg-size "${MSG_SIZE}"
  --flit-size "${FLIT_SIZE}"
  --inject-gap "${INJECT_GAP}"
  --out "traces/${CSV_BASENAME}"
  --hardware-metadata-json traces/noi_hbm_hardware_metadata.json
)
[[ "${SMOKE}" -eq 1 ]] && PYTORCHSIM_ARGS+=(--smoke)

echo "========== [1/3] PyTorchSim (Docker: ${DOCKER_IMAGE}) =========="
if [[ "${SKIP_PYTORCHSIM}" -eq 0 ]]; then
  docker run --rm --ipc=host \
    -v "${TORCHSIM_DIR}/scripts:/workspace/PyTorchSim/scripts:ro" \
    -v "${TOPOTRACE_ROOT}/traces:/workspace/PyTorchSim/traces" \
    -v "${TORCHSIM_DIR}/togsim_results:/workspace/PyTorchSim/togsim_results" \
    -w /workspace/PyTorchSim \
    "${DOCKER_IMAGE}" \
    python scripts/run_a2a_pytorchsim.py "${PYTORCHSIM_ARGS[@]}"
  # Docker writes to mounted TOPOTRACE_ROOT/traces
else
  echo "(skipped)"
fi

[[ -f "${CSV_TRACE}" ]] || { echo "Missing: ${CSV_TRACE}"; exit 1; }
CSV_DATA_ROWS=$(( $(wc -l < "${CSV_TRACE}") - 1 ))
echo "CSV: ${CSV_TRACE} (${CSV_DATA_ROWS} rows)"
if [[ "${SMOKE}" -eq 0 && "${CSV_DATA_ROWS}" -ne "${EXPECTED_EVENTS}" ]]; then
  echo "ERROR: expected ${EXPECTED_EVENTS} CSV rows, got ${CSV_DATA_ROWS}"
  exit 1
fi

echo "========== [2/3] CSV -> PopNet bench =========="
python3 "${SCRIPT_DIR}/pytorchsim_csv_to_popnet.py" \
  --in "${CSV_TRACE}" \
  --out-dir "${POPNET_TRACE_DIR}" \
  --nodes "${NODES}" \
  --mesh-ary "${TOPOTRACE_POPNET_MESH_ARY}"

BENCH_LINES=$(wc -l < "${POPNET_TRACE_DIR}/bench")
if [[ "${BENCH_LINES}" -ne "${EXPECTED_EVENTS}" ]]; then
  echo "ERROR: bench has ${BENCH_LINES} lines, expected ${EXPECTED_EVENTS}"
  exit 1
fi

[[ "${SKIP_POPNET}" -eq 1 ]] && { echo "Done (--skip-popnet)."; exit 0; }

echo "========== [3/3] PopNet =========="
if [[ ! -x "${POPNET_BIN}" ]]; then
  mkdir -p "${POPNET_DIR}/build"
  (cd "${POPNET_DIR}/build" && cmake .. && make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)")
fi

set +e
"${POPNET_BIN}" -A "${TOPOTRACE_POPNET_MESH_ARY}" -c "${TOPOTRACE_POPNET_MESH_DIMS}" \
  -V 3 -B 12 -O 12 -F 4 -L 1000 -T "${SIM_CYCLES}" -r 1 \
  -I "${POPNET_TRACE_DIR}/bench" -R 0 2>&1 | tee "${POPNET_LOG}"
POPNET_RC=${PIPESTATUS[0]}
set -e
[[ "${POPNET_RC}" -eq 0 ]] || exit "${POPNET_RC}"

grep -q "Packet count: ${EXPECTED_EVENTS}" "${POPNET_LOG}" || exit 1
grep -q "Finished packets: ${EXPECTED_EVENTS}" "${POPNET_LOG}" || exit 1

echo "========== TopoTraceSim pipeline OK =========="
echo "Hardware summary: ${HARDWARE_SUMMARY}"
