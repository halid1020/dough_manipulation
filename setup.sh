#!/bin/bash

echo "=== Starting Setup ==="

# 1. Initialize Conda and activate the environment
echo "[1/3] Activating conda environment: dough-manip..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dough-manip

# 2. Create the soft link for the dataset
echo "[2/3] Creating soft link for datasets..."
if [ -d "../DiffSkill/datasets" ]; then
    ln -sfn ../DiffSkill/datasets datasets
    echo "      -> Symlink created: datasets -> ../DiffSkill/datasets"
else
    echo "      -> Warning: Target directory ../DiffSkill/datasets does not exist yet."
    ln -sfn ../DiffSkill/datasets datasets
fi

# 3. Setting up Environment Variables (Replaces prepare.sh)
echo "[3/3] Exporting DiffSkill environment variables..."

# Check if DiffSkill exists before setting paths
if [ -d "../DiffSkill" ]; then
    # Get the absolute paths
    ROOT_DIR="$(pwd)"
    DIFFSKILL_DIR="$(cd ../DiffSkill && pwd)"
    
    # We include ROOT_DIR here to fix the "No module named 'registration'" error!
    export PYTHONPATH="${ROOT_DIR}:${DIFFSKILL_DIR}:${DIFFSKILL_DIR}/taichi_three:${DIFFSKILL_DIR}/PointFlow:${DIFFSKILL_DIR}/setvae:${PYTHONPATH}"
    
    # Exporting paths exactly as they were in prepare.sh
    export PATH="${DIFFSKILL_DIR}/taichi_three:${PATH}:/opt/cuda/11.1.1/bin"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/cuda/11.1.1/lib64"
    
    echo "      -> Variables exported successfully."
else
    echo "      -> Error: ../DiffSkill directory not found! Paths not exported."
    return 1
fi

echo "=== Setup Complete! ==="