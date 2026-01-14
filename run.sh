#!/usr/bin/env bash
set -e

# Resolve workspace root (directory of this script's parent, or current dir)
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)/bounding-expert-hierarchies"

export PYTHONPATH="${WORKSPACE_DIR}"

DATA_NAMES_2D=(
  dwarfumbrella
  sakura
  swisscheesevine
  rubber
  monstera
)

DATA_NAMES_3D=(
  apple_point_1M_grid_samples
  bunny_point_1M_grid_samples
  deer_point_1M_grid_samples
  jade_complex_point_1M_grid_samples
  jade_point_1M_grid_samples
  tree_point_1M_grid_samples
)

for data_name in "${DATA_NAMES_3D[@]}"; do
  echo "=== Running data_name=${data_name} ==="
  python "${WORKSPACE_DIR}/scripts/train_evaluate.py" \
    --data_name "${data_name}" \
    --query point \
    --dim 3
done