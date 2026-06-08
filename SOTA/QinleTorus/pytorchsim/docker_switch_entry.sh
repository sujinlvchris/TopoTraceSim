#!/usr/bin/env bash
# Run the google/switch-base-8 benchmark driver inside torchsim-ci.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QINLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DOCKER_IMAGE="${TORCHSIM_DOCKER_IMAGE:-ghcr.io/psal-postech/torchsim-ci:v1.0.0}"

TOPOTRACE_PYTORCHSIM_DIR="${TOPOTRACE_PYTORCHSIM_DIR:-}"
if [[ -z "${TOPOTRACE_PYTORCHSIM_DIR}" ]]; then
  for cand in \
      "${QINLE_ROOT}/../../TopoTraceSim/PyTorchSim" \
      "/mnt/sdb1/wyf/TopoTraceSim/PyTorchSim"; do
    if [[ -d "${cand}" ]]; then
      TOPOTRACE_PYTORCHSIM_DIR="$(cd "${cand}" && pwd)"
      break
    fi
  done
fi

mkdir -p "${QINLE_ROOT}/traces"

DOCKER_ARGS=(
  --rm --ipc=host
  -v "${QINLE_ROOT}/pytorchsim:/workspace/PyTorchSim/scripts/qinle:ro"
  -v "${QINLE_ROOT}/traces:/workspace/PyTorchSim/traces"
)

if [[ -n "${TOPOTRACE_PYTORCHSIM_DIR}" && -d "${TOPOTRACE_PYTORCHSIM_DIR}" ]]; then
  DOCKER_ARGS+=(
    -v "${TOPOTRACE_PYTORCHSIM_DIR}/configs:/workspace/PyTorchSim/configs:ro"
  )
  if [[ -d "${TOPOTRACE_PYTORCHSIM_DIR}/PyTorchSimBackend" ]]; then
    DOCKER_ARGS+=(
      -v "${TOPOTRACE_PYTORCHSIM_DIR}/PyTorchSimBackend:/workspace/PyTorchSim/PyTorchSimBackend:ro"
    )
  fi
  echo "Mounted upstream PyTorchSim configs from ${TOPOTRACE_PYTORCHSIM_DIR}"
fi

SWITCH_DRIVER="${SWITCH_DRIVER:-run_switch_moe_layer.py}"

DOCKER_ARGS+=(
  -w /workspace/PyTorchSim
  "${DOCKER_IMAGE}"
  python "scripts/qinle/${SWITCH_DRIVER}" "$@"
)

echo "docker run ${DOCKER_ARGS[*]}"
docker run "${DOCKER_ARGS[@]}"
