#!/usr/bin/env bash 
set -e # Exit bash script if a following command fails

# Script to automatically create a python enviornment based on the 
# requirements files within this repo and activate it

# ===== Configuration =====
VENV_DIR=".venv"
PYTHON_BIN="python3"
REQ_FILE="requirements.txt"

# ===== Create Virtual Environment =====
echo "▶ Creating virtual environment in ${VENV_DIR}"

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_BIN -m venv "$VENV_DIR"
else
    echo "✔ Virtual environment already exists"
fi

# ===== Activate Virtual Environment =====
echo "▶ Activating virtual environment"
source "$VENV_DIR/bin/activate"

# ===== Upgrade pip =====
echo "▶ Upgrading pip"
pip install --upgrade pip setuptools wheel

# ===== Install Requirements =====
if [ -f "$REQ_FILE" ]; then
    echo "▶ Installing requirements from ${REQ_FILE}"
    pip install -r "$REQ_FILE"
else
    echo "⚠ No requirements.txt found, skipping dependency install"
fi

# ===== Completion Status =====
echo "✔ Environment ready"
echo "Python: $(which python)"