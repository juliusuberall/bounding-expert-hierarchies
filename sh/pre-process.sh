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

DATA_NAMES_3D=(
  apple
  bunny
  deer
  jade_complex
  tree
)

DATA_NAMES_4D=(
  fluid_point_1to20_1f_1M_samples
  fluid_point_21to40_1f_1M_samples
  fluid_point_41to60_1f_1M_samples
  fluid_point_61to80_1f_1M_samples
  fluid_point_81to100_1f_1M_samples
)

for data_name in "${DATA_NAMES_3D[@]}"; do
    echo "=== Running data_name=${data_name} ==="
    python "${WORKSPACE_DIR}/scripts/preprocess.py" \
        --data_name "${data_name}" \
        --query ray \
        --dim 3 \
        --res 100 \
        --strategy grid \
        --start 81 \
        --end 100 \
        --frame_rate 2
done