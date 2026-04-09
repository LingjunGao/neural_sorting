#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}/examples"
DATA_DIR="${EXAMPLES_DIR}/data"
RESULTS_DIR="${EXAMPLES_DIR}/results"
DATASET_DIR="${DATA_DIR}/360_v2"

ZENODO_RECORD_ID="19420924"
ZENODO_FILES_BASE="https://zenodo.org/records/${ZENODO_RECORD_ID}/files"

REQUIRED_SCENES=(bicycle bonsai counter garden kitchen room)

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}"

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] python/python3 not found in PATH."
  exit 1
fi

have_dataset() {
  [[ -d "${DATASET_DIR}" ]] || return 1
  for s in "${REQUIRED_SCENES[@]}"; do
    [[ -d "${DATASET_DIR}/${s}" ]] || return 1
  done
  return 0
}

is_valid_tar_gz() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  tar -tzf "$f" >/dev/null 2>&1
}

download_file() {
  local url="$1"
  local out="$2"
  echo "[INFO] Downloading: ${url}"
  curl -fL --retry 5 --retry-delay 2 --connect-timeout 30 -C - -o "${out}" "${url}"
}

download_dataset_if_needed() {
  if have_dataset; then
    echo "[INFO] Dataset already exists at ${DATASET_DIR}. Skip download."
    return 0
  fi

  echo "[INFO] Downloading mip-NeRF360 dataset..."
  (
    cd "${EXAMPLES_DIR}"
    "${PYTHON_BIN}" datasets/download_dataset.py
  )
}

download_and_extract_checkpoint() {
  local canonical_archive="$1"
  local extract_dir_name="$2"
  local alt_archive="${3:-}"

  local canonical_path="${RESULTS_DIR}/${canonical_archive}"
  local alt_path=""
  [[ -n "${alt_archive}" ]] && alt_path="${RESULTS_DIR}/${alt_archive}"

  if [[ -d "${RESULTS_DIR}/${extract_dir_name}" ]]; then
    echo "[INFO] ${extract_dir_name} already extracted. Skip."
    return 0
  fi

  local archive_to_use=""
  if is_valid_tar_gz "${canonical_path}"; then
    archive_to_use="${canonical_path}"
    echo "[INFO] Found existing archive: ${canonical_path}"
  elif [[ -n "${alt_path}" ]] && is_valid_tar_gz "${alt_path}"; then
    archive_to_use="${alt_path}"
    echo "[INFO] Found existing archive: ${alt_path}"
  fi

  if [[ -z "${archive_to_use}" ]]; then
    rm -f "${canonical_path}"
    [[ -n "${alt_path}" ]] && rm -f "${alt_path}"

    if download_file "${ZENODO_FILES_BASE}/${canonical_archive}?download=1" "${canonical_path}"; then
      archive_to_use="${canonical_path}"
    elif [[ -n "${alt_archive}" ]]; then
      echo "[WARN] Canonical archive not found. Trying fallback: ${alt_archive}"
      download_file "${ZENODO_FILES_BASE}/${alt_archive}?download=1" "${alt_path}"
      archive_to_use="${alt_path}"
    else
      echo "[ERROR] Failed to download ${canonical_archive}"
      return 1
    fi
  fi

  if ! is_valid_tar_gz "${archive_to_use}"; then
    echo "[ERROR] Archive appears corrupted: ${archive_to_use}"
    return 1
  fi

  echo "[INFO] Extracting $(basename "${archive_to_use}") to ${RESULTS_DIR}"
  tar -xzf "${archive_to_use}" -C "${RESULTS_DIR}"
}

echo "================ Docker data/checkpoint setup ================"

download_dataset_if_needed

download_and_extract_checkpoint "benchmark.tar.gz" "benchmark" "bechmark.tar.gz"
download_and_extract_checkpoint "mlp_checkpoint.tar.gz" "mlp_checkpoint"

if [[ -d "${RESULTS_DIR}/non_clone" && ! -d "${RESULTS_DIR}/mlp_checkpoint" ]]; then
  echo "[INFO] Renaming ${RESULTS_DIR}/non_clone -> ${RESULTS_DIR}/mlp_checkpoint"
  mv "${RESULTS_DIR}/non_clone" "${RESULTS_DIR}/mlp_checkpoint"
fi

echo "[INFO] Done."
echo "[INFO] Dataset dir:   ${DATASET_DIR}"
echo "[INFO] Results dir:   ${RESULTS_DIR}"
echo "[INFO] Checkpoint dir: ${RESULTS_DIR}/mlp_checkpoint"
