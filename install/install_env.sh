#!/usr/bin/env bash

set -o errexit
set -o pipefail
set -o nounset

# Set the conda environment name
CONDA_ENV="scip-env-conda"

# Define the working directories
WORK_DIR="$(pwd)"
INSTALL_FOLDER="${WORK_DIR}/install"

# Create necessary directories if they do not exist
mkdir -p "${INSTALL_FOLDER}"

# Check if conda environment exists, if so, remove it
if conda info --envs | grep "${CONDA_ENV}" > /dev/null 2>&1; then
  echo "Conda environment ${CONDA_ENV} already exists. Removing it..."
  conda deactivate 2>/dev/null || true
  conda env remove -n "${CONDA_ENV}" -y
  echo "Conda environment removed."
fi

# Create conda environment from env.yaml
echo "Creating conda environment ${CONDA_ENV} from ${CONDA_ENV}.yaml..."
conda env create -f "${INSTALL_FOLDER}/${CONDA_ENV}.yaml"
echo "Conda environment created. "