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

SWITCH_MODEL_DIR="${SWITCH_MODEL_DIR:-}"
SWITCH_MODEL_CONTAINER_DIR="${SWITCH_MODEL_CONTAINER_DIR:-/workspace/hf_models/google_switch_base_8}"
if [[ -z "${SWITCH_MODEL_DIR}" ]]; then
  for cand in \
      "/mnt/sdb1/wyf/hf_models/google_switch_base_8" \
      "${QINLE_ROOT}/../../../../hf_models/google_switch_base_8"; do
    if [[ -d "${cand}" ]]; then
      SWITCH_MODEL_DIR="$(cd "${cand}" && pwd)"
      break
    fi
  done
fi

if [[ -n "${SWITCH_MODEL_DIR}" && -d "${SWITCH_MODEL_DIR}" ]]; then
  DOCKER_ARGS+=(
    -v "${SWITCH_MODEL_DIR}:${SWITCH_MODEL_CONTAINER_DIR}:ro"
    -e "SWITCH_MODEL_CONTAINER_DIR=${SWITCH_MODEL_CONTAINER_DIR}"
  )
  echo "Mounted Switch HF model from ${SWITCH_MODEL_DIR}"
fi

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
