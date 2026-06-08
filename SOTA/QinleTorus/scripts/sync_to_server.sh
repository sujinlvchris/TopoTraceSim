#!/usr/bin/env bash
# Sync QinleTorus to the Linux server used for PyTorchSim + PopNet runs.
#
# Usage:
#   bash scripts/sync_to_server.sh
#   REMOTE_DIR=/mnt/sdb1/wyf/SOTA/QinleTorus bash scripts/sync_to_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QINLE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SSH_PORT="${SSH_PORT:-9370}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ecdsa}"
REMOTE_USER="${REMOTE_USER:-wyf}"
REMOTE_HOST="${REMOTE_HOST:-10.98.36.113}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/sdb1/wyf/SOTA/QinleTorus}"

echo "sync ${QINLE_ROOT}/ -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

ssh -p "${SSH_PORT}" -i "${SSH_KEY}" \
  "${REMOTE_USER}@${REMOTE_HOST}" \
  "mkdir -p '${REMOTE_DIR}'"

rsync -avz \
  -e "ssh -p ${SSH_PORT} -i ${SSH_KEY}" \
  --exclude '.DS_Store' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'output/run_*/' \
  "${QINLE_ROOT}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

echo "sync done"
