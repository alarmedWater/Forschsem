#!/usr/bin/env bash
# pipeline_detectron2.sh – Env für Detectron2 v0.6 anlegen und Mask R-CNN trainieren
# Beispiele:
#   ./pipeline_detectron2.sh
#   ./pipeline_detectron2.sh --cuda
#   ./pipeline_detectron2.sh --force

set -euo pipefail

PY310="3.10"
MODE="cpu"
FORCE="no"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE="cpu"; shift ;;
    --cuda) MODE="cuda"; shift ;;   # Hinweis: Wheel unten ist CPU
    --force) FORCE="yes"; shift ;;
    -h|--help)
      echo "Usage: $0 [--cpu|--cuda] [--force]"; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

ENV_NAME="detectron2-06"
echo "[D2] MODE=${MODE}, FORCE=${FORCE}"

create_or_reuse () {
  local NAME="$1"
  local PYVER="$2"
  if conda env list | awk '{print $1}' | grep -qx "${NAME}"; then
    if [[ "${FORCE}" == "yes" ]]; then
      echo "[D2] Removing existing env '${NAME}' (force)..."
      conda env remove -n "${NAME}" -y || true
    else
      echo "[D2] Env '${NAME}' exists. Reusing."
      return
    fi
  fi
  echo "[D2] Creating env '${NAME}' (Python ${PYVER})..."
  conda create -y -n "${NAME}" python="${PYVER}"
}

install_pytorch_stack () {
  local NAME="$1"
  # Detectron2 0.6 -> Torch 1.13.1 / TV 0.14.1
  if [[ "${MODE}" == "cuda" ]]; then
    # hier müsstest du ein CUDA-kompatibles D2 selbst bauen
    conda install -y -n "${NAME}" -c pytorch -c nvidia \
      pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.7
  else
    conda install -y -n "${NAME}" -c pytorch \
      pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 cpuonly
  fi
  conda install -y -n "${NAME}" -c conda-forge "intel-openmp>=2021" "_openmp_mutex=*=*intel" || true
  conda remove -y -n "${NAME}" llvm-openmp || true
}

base_science_stack () {
  local NAME="$1"
  conda install -y -n "${NAME}" -c conda-forge \
    "numpy<2" pandas matplotlib scikit-image scikit-learn pillow tqdm pyyaml cython tabulate rich
}

create_or_reuse "${ENV_NAME}" "${PY310}"
install_pytorch_stack "${ENV_NAME}"
base_science_stack "${ENV_NAME}"
conda install -y -n "${ENV_NAME}" -c conda-forge opencv termcolor

echo "[D2] Installing Detectron2 v0.6 (CPU wheel)..."
conda run -n "${ENV_NAME}" pip install --no-cache-dir \
  "detectron2==0.6" -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.13/index.html

echo "[D2] Running training script..."
conda run -n "${ENV_NAME}" python train_detectron2.py

echo "[D2] Done ✅"
