#!/usr/bin/env bash
# End-to-end: PyTorchSim (Gem5+Spike+BackendSim/TOGSim) -> CSV -> PopNet (2x2 mesh)
#
# Server example:
#   export POPNET_DIR=/mnt/sdb1/wyf/popnet_anytopo
#   cd /mnt/sdb1/wyf/PyTorchSim
#   bash scripts/run_a2a_full_pipeline.sh
#
# Smoke (1 matmul + skip full PopNet check):
#   bash scripts/run_a2a_full_pipeline.sh --smoke

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHSIM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
POPNET_DIR="${POPNET_DIR:-$(cd "${TORCHSIM_DIR}/.." && pwd)/popnet_anytopo}"

DOCKER_IMAGE="${TORCHSIM_DOCKER_IMAGE:-ghcr.io/psal-postech/torchsim-ci:v1.0.0}"
NODES=4
MSG_SIZE=16KB
FLIT_SIZE=64
INJECT_GAP=0
SIM_CYCLES=100000

CSV_TRACE="${TORCHSIM_DIR}/traces/a2a_n4_16kb_pytorchsim.csv"
POPNET_TRACE_DIR="${TORCHSIM_DIR}/popnet_exp/traces/a2a_2x2"
POPNET_LOG="${TORCHSIM_DIR}/popnet_exp/logs/a2a_2x2_run.log"
POPNET_BIN="${POPNET_DIR}/build/popnet"

SMOKE=0
SKIP_PYTORCHSIM=0
SKIP_POPNET=0
CONVERT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --skip-pytorchsim) SKIP_PYTORCHSIM=1; shift ;;
    --skip-popnet) SKIP_POPNET=1; shift ;;
    --convert-only) CONVERT_ONLY=1; SKIP_PYTORCHSIM=1; shift ;;
    --popnet-dir) POPNET_DIR="$2"; POPNET_BIN="${POPNET_DIR}/build/popnet"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

EXPECTED_EVENTS=$((NODES * (NODES - 1)))
[[ "${SMOKE}" -eq 1 ]] && EXPECTED_EVENTS=1

mkdir -p "${TORCHSIM_DIR}/traces" "${TORCHSIM_DIR}/popnet_exp/logs" "${POPNET_TRACE_DIR}"

PYTORCHSIM_ARGS=(
  --nodes "${NODES}"
  --msg-size "${MSG_SIZE}"
  --flit-size "${FLIT_SIZE}"
  --inject-gap "${INJECT_GAP}"
  --out "traces/a2a_n4_16kb_pytorchsim.csv"
)
[[ "${SMOKE}" -eq 1 ]] && PYTORCHSIM_ARGS+=(--smoke)

echo "========== [1/3] PyTorchSim (Docker: ${DOCKER_IMAGE}) =========="
if [[ "${SKIP_PYTORCHSIM}" -eq 0 ]]; then
  docker run --rm --ipc=host \
    -v "${TORCHSIM_DIR}/scripts:/workspace/PyTorchSim/scripts:ro" \
    -v "${TORCHSIM_DIR}/traces:/workspace/PyTorchSim/traces" \
    -v "${TORCHSIM_DIR}/togsim_results:/workspace/PyTorchSim/togsim_results" \
    -w /workspace/PyTorchSim \
    "${DOCKER_IMAGE}" \
    python scripts/run_a2a_pytorchsim.py "${PYTORCHSIM_ARGS[@]}"
else
  echo "(skipped --skip-pytorchsim)"
fi

[[ -f "${CSV_TRACE}" ]] || { echo "Missing PyTorchSim CSV: ${CSV_TRACE}"; exit 1; }

CSV_DATA_ROWS=$(( $(wc -l < "${CSV_TRACE}") - 1 ))
echo "PyTorchSim CSV: ${CSV_TRACE} (${CSV_DATA_ROWS} data rows)"
if [[ "${SMOKE}" -eq 0 && "${CSV_DATA_ROWS}" -ne "${EXPECTED_EVENTS}" ]]; then
  echo "ERROR: CSV has ${CSV_DATA_ROWS} rows, expected ${EXPECTED_EVENTS}."
  echo "  Re-run step 1 without --smoke, or: bash scripts/run_a2a_full_pipeline.sh"
  exit 1
fi

echo "========== [2/3] PyTorchSim CSV -> PopNet bench =========="
python3 "${SCRIPT_DIR}/pytorchsim_csv_to_popnet.py" \
  --in "${CSV_TRACE}" \
  --out-dir "${POPNET_TRACE_DIR}"

[[ -f "${POPNET_TRACE_DIR}/bench" ]] || { echo "Missing PopNet bench"; exit 1; }
BENCH_LINES=$(wc -l < "${POPNET_TRACE_DIR}/bench")
echo "PopNet bench: ${POPNET_TRACE_DIR}/bench (${BENCH_LINES} lines)"
if [[ "${BENCH_LINES}" -ne "${EXPECTED_EVENTS}" ]]; then
  echo "ERROR: bench has ${BENCH_LINES} lines, expected ${EXPECTED_EVENTS}."
  echo "  Stale bench from an old --smoke run? Re-run conversion:"
  echo "  python3 ${SCRIPT_DIR}/pytorchsim_csv_to_popnet.py"
  exit 1
fi

if [[ "${SKIP_POPNET}" -eq 1 ]]; then
  echo "Done (--skip-popnet)."
  exit 0
fi

echo "========== [3/3] PopNet 2x2 fixed mesh =========="
if [[ ! -x "${POPNET_BIN}" ]]; then
  echo "Building PopNet via CMake in ${POPNET_DIR}/build ..."
  mkdir -p "${POPNET_DIR}/build"
  (cd "${POPNET_DIR}/build" && cmake .. && make -j"$(nproc)")
fi

POPNET_ARGS=(
  -A 2 -c 2 -V 3 -B 12 -O 12 -F 4
  -L 1000 -T "${SIM_CYCLES}" -r 1
  -I "${POPNET_TRACE_DIR}/bench"
  -R 0
)

set +e
"${POPNET_BIN}" "${POPNET_ARGS[@]}" 2>&1 | tee "${POPNET_LOG}"
POPNET_RC=${PIPESTATUS[0]}
set -e

[[ "${POPNET_RC}" -eq 0 ]] || { echo "PopNet failed (exit ${POPNET_RC})"; exit "${POPNET_RC}"; }

if ! grep -q "Packet count: ${EXPECTED_EVENTS}" "${POPNET_LOG}"; then
  echo "PopNet did not report Packet count: ${EXPECTED_EVENTS}"
  exit 1
fi
if ! grep -q "Finished packets: ${EXPECTED_EVENTS}" "${POPNET_LOG}"; then
  echo "PopNet did not finish all packets (expected ${EXPECTED_EVENTS})"
  exit 1
fi

echo "========== Pipeline OK =========="
echo "  PyTorchSim CSV : ${CSV_TRACE}"
echo "  PopNet trace   : ${POPNET_TRACE_DIR}/bench"
echo "  PopNet log     : ${POPNET_LOG}"
