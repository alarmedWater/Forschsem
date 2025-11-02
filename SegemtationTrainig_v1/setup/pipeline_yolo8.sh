#!/usr/bin/env bash
# pipeline_yolo8_setup.sh – legt ein sauberes YOLOv8-Env an (ohne Training)
# Beispiele:
#   ./pipeline_yolo8_setup.sh
#   ./pipeline_yolo8_setup.sh --cuda
#   ./pipeline_yolo8_setup.sh --force

set -euo pipefail

PY310="3.10"
MODE="cpu"     # cpu | cuda
FORCE="no"
ENV_NAME="yolo8-seg"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE="cpu"; shift ;;
    --cuda) MODE="cuda"; shift ;;
    --force) FORCE="yes"; shift ;;
    -h|--help)
      echo "Usage: $0 [--cpu|--cuda] [--force]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

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
  conda create -y -n "${NAME}" -c conda-forge python="${PYVER}"
}

install_pytorch_stack () {
  local NAME="$1"

  if [[ "${MODE}" == "cuda" ]]; then
    # falls du später echte GPU hast – sonst einfach --cpu nutzen
    conda install -y -n "${NAME}" -c pytorch -c nvidia \
      pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=12.1
  else
    # exakt die Kombi, die jetzt bei dir läuft
    conda install -y -n "${NAME}" -c pytorch \
      pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 cpuonly
  fi
}

base_science_stack () {
  local NAME="$1"
  conda install -y -n "${NAME}" -c conda-forge \
    "numpy<2" pandas pyyaml opencv pillow requests
}

create_or_reuse "${ENV_NAME}" "${PY310}"
install_pytorch_stack "${ENV_NAME}"
base_science_stack "${ENV_NAME}"

echo "[YOLOv8] Installing pip deps (Ultralytics & ONNX)..."
conda run -n "${ENV_NAME}" pip install --no-cache-dir -U \
  "ultralytics==8.3.223" \
  "pycocotools>=2.0.0,<3.0.0" \
  "onnx==1.19.1" \
  "onnxruntime==1.16.3" \
  "onnxslim==0.1.72" \
  "albumentations==1.4.24" \
  "thop==0.1.1.post2209072238"

echo
echo "[YOLOv8] ✅ Environment '${ENV_NAME}' is ready."
echo "Use it with:"
echo "  conda activate ${ENV_NAME}"
echo "  python /home/parallels/Forschsemrep/SegemtationTrainig_v1/models/train_yolo8s.py"
