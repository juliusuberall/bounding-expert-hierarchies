#!/usr/bin/env bash
set -e

# Resolve workspace root as the current directory
WORKSPACE_DIR="$(pwd)"
export PYTHONPATH="${WORKSPACE_DIR}"

DATA_NAMES_2D=(
  dwarfumbrella
  sakura
  swisscheesevine
  rubber
  monstera
)

for data_name in "${DATA_NAMES_2D[@]}"; do
  echo "=== Running data_name=${data_name} ==="
  python "${WORKSPACE_DIR}/scripts/train_evaluate.py" \
    --data_name "${data_name}" \
    --query point \
    --dim 2
done