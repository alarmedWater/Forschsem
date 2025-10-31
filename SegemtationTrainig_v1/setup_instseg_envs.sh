#!/usr/bin/env bash
# setup_instseg_envs.sh — erstellt getrennte Conda-Envs:
#   - yolo8-seg     (Ultralytics/YOLOv8, Torch 2.1.x)
#   - detectron2-06 (Detectron2 v0.6, Torch 1.13.1, CPU-Wheel)
#   - yolactpp      (YOLACT++, Torch 1.10.2)
# Beispiele:
#   ./setup_instseg_envs.sh --all
#   ./setup_instseg_envs.sh --target yolo8
#   ./setup_instseg_envs.sh --target d2
#   ./setup_instseg_envs.sh --target yolactpp --force
#   ./setup_instseg_envs.sh --target d2 --cuda   # (Hinweis: D2-Wheel unten ist CPU; CUDA für D2 bitte gesondert pinnen)

set -euo pipefail

PY310="3.10"
PY38="3.8"
MODE="cpu"          # cpu | cuda
FORCE="no"
TARGET="all"        # all | yolo8 | d2 | yolactpp

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) MODE="cpu"; shift ;;
    --cuda) MODE="cuda"; shift ;;
    --force) FORCE="yes"; shift ;;
    --target) TARGET="$2"; shift 2 ;;
    --all) TARGET="all"; shift ;;
    -h|--help)
      grep '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "[INFO] MODE=${MODE}, FORCE=${FORCE}, TARGET=${TARGET}"

create_or_reuse () {
  local NAME="$1"
  local PYVER="$2"

  if conda env list | awk '{print $1}' | grep -qx "${NAME}"; then
    if [[ "${FORCE}" == "yes" ]]; then
      echo "[INFO] Removing existing env '${NAME}' (force)..."
      conda env remove -n "${NAME}" -y || true
    else
      echo "[INFO] Env '${NAME}' exists. Reusing."
      return
    fi
  fi
  echo "[INFO] Creating env '${NAME}' (Python ${PYVER})..."
  conda create -y -n "${NAME}" python="${PYVER}"
}

install_pytorch_stack () {
  local NAME="$1"
  local TORCH="$2"
  local TV="$3"
  local TA="$4"

  if [[ "${MODE}" == "cuda" ]]; then
    echo "[INFO] Installing CUDA PyTorch stack into '${NAME}'..."
    conda install -y -n "${NAME}" -c pytorch -c nvidia \
      pytorch="${TORCH}" torchvision="${TV}" torchaudio="${TA}" pytorch-cuda=12.1
  else
    echo "[INFO] Installing CPU-only PyTorch stack into '${NAME}'..."
    conda install -y -n "${NAME}" -c pytorch \
      pytorch="${TORCH}" torchvision="${TV}" torchaudio="${TA}" cpuonly
  fi

  # --- OpenMP/ITT Fix: erzeuge Intel-OpenMP Mutex, entferne LLVM-OpenMP ---
  conda install -y -n "${NAME}" -c conda-forge "intel-openmp>=2021" "_openmp_mutex=*=*intel" || true
  conda remove -y -n "${NAME}" llvm-openmp || true
}

base_science_stack () {
  local NAME="$1"
  echo "[INFO] Installing base SciPy stack into '${NAME}'..."
  conda install -y -n "${NAME}" -c conda-forge \
    "numpy<2" pandas matplotlib scikit-image scikit-learn pillow tqdm pyyaml cython tabulate rich
}

finish_check () {
  local NAME="$1"
  echo "[INFO] Verifying '${NAME}'..."
  conda run -n "${NAME}" python - <<'PY'
import sys
print("Python:", sys.version.split()[0])
try:
    import numpy, torch
    print("NumPy:", numpy.__version__)
    print("Torch:", torch.__version__)
except Exception as e:
    print("Core import error:", e); sys.exit(1)
print("OK")
PY
}

# ------------------ YOLOv8 Env ------------------
setup_yolo8 () {
  local NAME="yolo8-seg"
  create_or_reuse "${NAME}" "${PY310}"
  install_pytorch_stack "${NAME}" "2.1.2" "0.16.2" "2.1.2"
  base_science_stack "${NAME}"

  # OpenCV stabil via conda (kein pip-OpenCV zusätzlich!)
  conda install -y -n "${NAME}" -c conda-forge opencv

  echo "[INFO] Installing YOLOv8 & friends (pip) into '${NAME}'..."
  conda run -n "${NAME}" pip install --no-cache-dir -U \
    "ultralytics>=8.1.0,<9.0.0" \
    "pycocotools>=2.0.0,<3.0.0" \
    "onnx>=1.13.0,<1.16.0" \
    "onnxruntime>=1.14.0,<1.17.0" \
    "onnxslim>=0.1.71,<0.2.0" \
    "albumentations>=1.3.0,<2.0.0" \
    "seaborn<0.14.0" \
    "psutil>=5.9.0,<6.0.0"
  # thop (optional) – kompatible Version:
  conda run -n "${NAME}" pip install "thop==0.1.1.post2209072238" || true

  finish_check "${NAME}"
}

# ------------------ Detectron2 v0.6 Env ------------------
setup_d2 () {
  local NAME="detectron2-06"
  create_or_reuse "${NAME}" "${PY310}"
  # D2 v0.6 passt zu Torch 1.13.1 / TV 0.14.1
  install_pytorch_stack "${NAME}" "1.13.1" "0.14.1" "0.13.1"
  base_science_stack "${NAME}"
  conda install -y -n "${NAME}" -c conda-forge opencv termcolor

  echo "[INFO] Installing Detectron2 v0.6 (prebuilt wheel for CPU, Torch 1.13.x)..."
  # WHEEL statt Source-Build: stabil & kein Compiler nötig
  conda run -n "${NAME}" pip install --no-cache-dir \
    "detectron2==0.6" -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.13/index.html

  # kurzer Import-Check inkl. detectron2
  conda run -n "${NAME}" python - <<'PY'
import torch, detectron2
print("Torch:", torch.__version__, "| D2 OK")
PY

  finish_check "${NAME}"
}

# ------------------ YOLACT++ Env ------------------
setup_yolactpp () {
  local NAME="yolactpp"
  create_or_reuse "${NAME}" "${PY38}"
  # YOLACT++ stabil mit Torch 1.10.2 / TV 0.11.3
  install_pytorch_stack "${NAME}" "1.10.2" "0.11.3" "0.10.2"
  base_science_stack "${NAME}"
  conda install -y -n "${NAME}" -c conda-forge opencv

  # Repo-spezifische Abhängigkeiten — bei Bedarf anpassen:
  conda run -n "${NAME}" pip install --no-cache-dir -U \
    "pycocotools>=2.0.0,<3.0.0" "numpy<2" "tqdm" "Pillow" "matplotlib" "scipy"
  # Optional: requirements des eigenen Forks
  # conda run -n "${NAME}" pip install -r /pfad/zu/yolactpp/requirements.txt || true

  finish_check "${NAME}"
}

# Dispatcher
case "${TARGET}" in
  all)      setup_yolo8; setup_d2; setup_yolactpp ;;
  yolo8)    setup_yolo8 ;;
  d2)       setup_d2 ;;
  yolactpp) setup_yolactpp ;;
  *) echo "Unknown target '${TARGET}'"; exit 1 ;;
esac

echo "===============================================================================
DONE ✅  Environments sind bereit.

Activate examples:
  conda activate yolo8-seg
  conda activate detectron2-06
  conda activate yolactpp

Quick tests:
  # YOLOv8
  conda run -n yolo8-seg python -c \"from ultralytics import YOLO; print('YOLOv8 OK')\"
  # Detectron2
  conda run -n detectron2-06 python -c \"from detectron2 import model_zoo; print('D2 OK')\"
  # OpenCV in YOLACT++
  conda run -n yolactpp python -c \"import cv2; print('OpenCV', cv2.__version__, 'OK')\"
===============================================================================
"
