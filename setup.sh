#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   source ./setup.sh
#
# If you run it with "bash ./setup.sh", installation still works,
# but conda environment activation cannot persist in your current shell.

METHOD_NAME="neural sorting"
ENV_NAME="neural_sorting"
PYTHON_VERSION="3.10"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"

is_sourced=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  is_sourced=1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found. Please install Miniconda/Anaconda first."
  if [[ ${is_sourced} -eq 1 ]]; then
    return 1
  else
    exit 1
  fi
fi

# Ensure conda activate works inside script
eval "$(conda shell.bash hook)"

echo "[1/7] Installing OS packages (requires sudo)..."
if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git wget build-essential gcc-11 g++-11 curl
else
  apt-get update
  apt-get install -y git wget build-essential gcc-11 g++-11 curl
fi

echo "[2/7] Creating conda environment (${ENV_NAME}) if needed..."
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" -y python="${PYTHON_VERSION}"
fi

echo "[3/7] Activating conda environment (${ENV_NAME})..."
conda activate "${ENV_NAME}"

echo "[4/7] Installing Python/CUDA dependencies..."
python -m pip install -U pip

echo "Checking for CUDA 11.8..."
CUDA_118_PATH=""
if [[ -d "/usr/local/cuda-11.8" ]]; then
  CUDA_118_PATH="/usr/local/cuda-11.8"
  echo "Found CUDA 11.8 at: $CUDA_118_PATH"
else
  if [[ -d "${CONDA_PREFIX}/lib/nvvm" ]]; then
    CUDA_118_PATH="${CONDA_PREFIX}"
    echo "Found CUDA in conda environment"
  else
    echo "CUDA 11.8 not found. Installing via conda..."
    conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
    CUDA_118_PATH="${CONDA_PREFIX}"
  fi
fi

if [[ -d "$CUDA_118_PATH" ]]; then
  export PATH="${CUDA_118_PATH}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_118_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  export CUDA_HOME="${CUDA_118_PATH}"
  echo "Set CUDA_HOME to: ${CUDA_118_PATH}"
fi

echo "Installing PyTorch for CUDA 11.8..."
python -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-cache-dir "numpy==1.26.4" "setuptools==69.5.1" "wheel<0.43" ninja packaging

echo "[5/7] Installing ${METHOD_NAME} from source..."
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11
python -m pip install -e "${SCRIPT_DIR}" --no-build-isolation

echo "[6/7] Installing examples requirements and downloading dataset..."
python -m pip install -r "${EXAMPLES_DIR}/requirements.txt" --no-build-isolation
(
  cd "${EXAMPLES_DIR}"
  python datasets/download_dataset.py
)

echo "[7/7] Optional benchmark download (Zenodo)"
RESULTS_DIR="${SCRIPT_DIR}/examples/results"
BENCHMARK_URL=""
BENCHMARK_ARCHIVE="${RESULTS_DIR}/mlp_checkpoint.tar.gz"

if [[ -n "${BENCHMARK_URL}" ]]; then
  mkdir -p "${RESULTS_DIR}"
  echo "Downloading mlp_checkpoint.tar.gz to ${RESULTS_DIR} ..."
  curl -L --progress-bar --retry 5 -o "${BENCHMARK_ARCHIVE}" "${BENCHMARK_URL}"
  echo "Extracting archive to ${RESULTS_DIR} ..."
  tar -xzf "${BENCHMARK_ARCHIVE}" -C "${RESULTS_DIR}"
  echo "Removing archive ..."
  rm -f "${BENCHMARK_ARCHIVE}"
  echo "mlp_checkpoint extracted to ${RESULTS_DIR}"
else
  echo "BENCHMARK_URL is empty. Skip benchmark download for now."
fi

echo ""
echo "Setup complete for ${METHOD_NAME}."
if [[ ${is_sourced} -eq 1 ]]; then
  echo "Conda environment '${ENV_NAME}' is active in this shell."
else
  echo "Conda environment activation cannot persist from a child shell."
  echo "Run this to activate it now: conda activate ${ENV_NAME}"
  echo "Tip: use 'source ./setup.sh' next time."
fi
