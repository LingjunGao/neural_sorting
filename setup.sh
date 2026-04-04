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

echo "[7/7] Downloading benchmark checkpoints from Zenodo"
RESULTS_DIR="${SCRIPT_DIR}/examples/results"
ZENODO_RECORD_ID="19420924"
ZENODO_FILE_BASE_URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files"

mkdir -p "${RESULTS_DIR}"

download_and_extract() {
  local archive_name="$1"
  local archive_url="$2"
  local extract_dir_hint="$3"
  local archive_path="${RESULTS_DIR}/${archive_name}"

  if [[ -d "${RESULTS_DIR}/${extract_dir_hint}" ]]; then
    echo "${extract_dir_hint} already exists under ${RESULTS_DIR}, skipping download."
    return 0
  fi

  echo "Downloading ${archive_name} ..."
  curl -fL --progress-bar --retry 5 -o "${archive_path}" "${archive_url}"
  echo "Extracting ${archive_name} to ${RESULTS_DIR} ..."
  tar -xzf "${archive_path}" -C "${RESULTS_DIR}"
  rm -f "${archive_path}"
}

# Some uploads used a typo (bechmark.tar.gz). Try benchmark first, then fallback.
if ! download_and_extract "benchmark.tar.gz" "${ZENODO_FILE_BASE_URL}/benchmark.tar.gz?download=1" "benchmark"; then
  echo "benchmark.tar.gz not found, trying bechmark.tar.gz ..."
  download_and_extract "bechmark.tar.gz" "${ZENODO_FILE_BASE_URL}/bechmark.tar.gz?download=1" "benchmark"
fi

download_and_extract "mlp_checkpoint.tar.gz" "${ZENODO_FILE_BASE_URL}/mlp_checkpoint.tar.gz?download=1" "mlp_checkpoint"

echo "Checkpoint archives extracted under ${RESULTS_DIR}."

echo ""
echo "Setup complete for ${METHOD_NAME}."
if [[ ${is_sourced} -eq 1 ]]; then
  echo "Conda environment '${ENV_NAME}' is active in this shell."
else
  echo "Conda environment activation cannot persist from a child shell."
  echo "Run this to activate it now: conda activate ${ENV_NAME}"
  echo "Tip: use 'source ./setup.sh' next time."
fi
