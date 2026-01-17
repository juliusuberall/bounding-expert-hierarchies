#!/usr/bin/env bash
set -e

# Resolve workspace root (directory of this script's parent, or current dir)
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

export PYTHONPATH="${WORKSPACE_DIR}"

DATA_NAMES_2D=(
  dwarfumbrella
  sakura
  swisscheesevine
  rubber
  monstera
)

DATA_NAMES_2D_RAYS=(
  dwarfumbrella_ray_1M_samples
  sakura_ray_1M_samples
  swisscheesevine_ray_1M_samples
  rubber_ray_1M_samples
  monstera_ray_1M_samples
)

DATA_NAMES_3D=(
  apple_point_1M_grid_samples
  bunny_point_1M_grid_samples
  deer_point_1M_grid_samples
  jade_complex_point_1M_grid_samples
  tree_point_1M_grid_samples
)

DATA_NAMES_3D_RAYS=(
  apple_ray_1M_samples
  bunny_ray_1M_samples
  deer_ray_1M_samples
  jade_complex_ray_1M_samples
  tree_ray_1M_samples
)

DATA_NAMES_4D=(
  fluid_point_1to20_1f_1M_samples
  fluid_point_21to40_1f_1M_samples
  fluid_point_41to60_1f_1M_samples
  fluid_point_61to80_1f_1M_samples
  fluid_point_81to100_1f_1M_samples
)

for data_name in "${DATA_NAMES_4D[@]}"; do
  echo "=== Running data_name=${data_name} ==="
  python "${WORKSPACE_DIR}/scripts/train_evaluate.py" \
    --data_name "${data_name}" \
    --query point \
    --dim 4
done