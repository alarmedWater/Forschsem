#!/usr/bin/env bash
# pipeline_yolactpp.sh – Env für YOLACT++ anlegen und Training starten
# Beispiele:
#   ./pipeline_yolactpp.sh
#   ./pipeline_yolactpp.sh --cuda
#   ./pipeline_yolactpp.sh --force

set -euo pipefail

PY38="3.8"    # YOLACT++ ist oft entspannter mit 3.8
MODE="cpu"
FORCE="no"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE="cpu"; shift ;;
    --cuda) MODE="cuda"; shift ;;
    --force) FORCE="yes"; shift ;;
    -h|--help)
      echo "Usage: $0 [--cpu|--cuda] [--force]"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

ENV_NAME="yolactpp"
echo "[YOLACT++] MODE=${MODE}, FORCE=${FORCE}"

create_or_reuse () {
  local NAME="$1"
  local PYVER="$2"
  if conda env list | awk '{print $1}' | grep -qx "${NAME}"; then
    if [[ "${FORCE}" == "yes" ]]; then
      echo "[YOLACT++] Removing existing env '${NAME}' (force)..."
      conda env remove -n "${NAME}" -y || true
    else
      echo "[YOLACT++] Env '${NAME}' exists. Reusing."
      return
    fi
  fi
  echo "[YOLACT++] Creating env '${NAME}' (Python ${PYVER})..."
  conda create -y -n "${NAME}" python="${PYVER}"
}

install_pytorch_stack () {
  local NAME="$1"
  # YOLACT++ stabil mit Torch 1.10.2 / TV 0.11.3
  if [[ "${MODE}" == "cuda" ]]; then
    conda install -y -n "${NAME}" -c pytorch -c nvidia \
      pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cudatoolkit=11.3
  else
    conda install -y -n "${NAME}" -c pytorch \
      pytorch=1.10.2 torchvision=0.11.3 torchaudio=0.10.2 cpuonly
  fi
  conda install -y -n "${NAME}" -c conda-forge "intel-openmp>=2021" "_openmp_mutex=*=*intel" || true
  conda remove -y -n "${NAME}" llvm-openmp || true
}

base_science_stack () {
  local NAME="$1"
  conda install -y -n "${NAME}" -c conda-forge \
    "numpy<2" pandas matplotlib scikit-image scikit-learn pillow tqdm pyyaml cython tabulate rich
}

create_or_reuse "${ENV_NAME}" "${PY38}"
install_pytorch_stack "${ENV_NAME}"
base_science_stack "${ENV_NAME}"
conda install -y -n "${ENV_NAME}" -c conda-forge opencv

echo "[YOLACT++] Installing extra pip deps..."
conda run -n "${ENV_NAME}" pip install --no-cache-dir -U \
  "pycocotools>=2.0.0,<3.0.0" "numpy<2" "tqdm" "Pillow" "matplotlib" "scipy"

# falls dein yolact++-Repo lokal ist, hier noch:
# conda run -n "${ENV_NAME}" pip install -e /pfad/zu/deinem/yolactpp || true

echo "[YOLACT++] Running training script..."
conda run -n "${ENV_NAME}" python train_yolactpp.py

echo "[YOLACT++] Done ✅"
