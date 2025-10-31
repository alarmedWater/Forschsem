#!/usr/bin/env bash
# pipeline_yolo8.sh – Env für YOLOv8-Seg anlegen und Training starten
# Beispiele:
#   ./pipeline_yolo8.sh
#   ./pipeline_yolo8.sh --cuda
#   ./pipeline_yolo8.sh --force

set -euo pipefail

PY310="3.10"
MODE="cpu"     # cpu | cuda
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

ENV_NAME="yolo8-seg"
echo "[YOLOv8] MODE=${MODE}, FORCE=${FORCE}"

create_or_reuse () {
  local NAME="$1"
  local PYVER="$2"

  if conda env list | awk '{print $1}' | grep -qx "${NAME}"; then
    if [[ "${FORCE}" == "yes" ]]; then
      echo "[YOLOv8] Removing existing env '${NAME}' (force)..."
      conda env remove -n "${NAME}" -y || true
    else
      echo "[YOLOv8] Env '${NAME}' exists. Reusing."
      return
    fi
  fi
  echo "[YOLOv8] Creating env '${NAME}' (Python ${PYVER})..."
  conda create -y -n "${NAME}" python="${PYVER}"
}

install_pytorch_stack () {
  local NAME="$1"
  if [[ "${MODE}" == "cuda" ]]; then
    conda install -y -n "${NAME}" -c pytorch -c nvidia \
      pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1
  else
    conda install -y -n "${NAME}" -c pytorch \
      pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 cpuonly
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
conda install -y -n "${ENV_NAME}" -c conda-forge opencv

echo "[YOLOv8] Installing pip deps..."
conda run -n "${ENV_NAME}" pip install --no-cache-dir -U \
  "ultralytics>=8.1.0,<9.0.0" \
  "pycocotools>=2.0.0,<3.0.0" \
  "onnx>=1.13.0,<1.16.0" \
  "onnxruntime>=1.14.0,<1.17.0" \
  "onnxslim>=0.1.71,<0.2.0" \
  "albumentations>=1.3.0,<2.0.0" \
  "seaborn<0.14.0" \
  "psutil>=5.9.0,<6.0.0"
conda run -n "${ENV_NAME}" pip install "thop==0.1.1.post2209072238" || true

echo "[YOLOv8] Running training script..."
conda run -n "${ENV_NAME}" python train_yolov8.py

echo "[YOLOv8] Done ✅"
