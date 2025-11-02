#!/bin/bash
# run_detectron2_docker.sh – Run Detectron2 in Docker

set -euo pipefail

MODE="${1:-cpu}"  # cpu or gpu
DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DOCKER_DIR/../.." && pwd)"

echo "[DOCKER] Building Docker image from: $DOCKER_DIR"
echo "[DOCKER] Project root: $PROJECT_ROOT"

# Check if required files exist
if [[ ! -f "$DOCKER_DIR/Dockerfile" ]]; then
    echo "[ERROR] Dockerfile not found at: $DOCKER_DIR/Dockerfile"
    exit 1
fi

if [[ ! -f "$DOCKER_DIR/train_detectron2.py" ]]; then
    echo "[ERROR] Training script not found at: $DOCKER_DIR/train_detectron2.py"
    exit 1
fi

# Build the image
echo "[DOCKER] Building Docker image (this may take a few minutes)..."
cd "$DOCKER_DIR"
docker build -t detectron2-training .

if [[ "$MODE" == "gpu" ]]; then
    echo "[DOCKER] Running with GPU support..."
    docker run --gpus all -it \
        -v "$PROJECT_ROOT:/workspace" \
        -w /workspace \
        -e DETECTRON2_USE_CUDA=1 \
        detectron2-training \
        python docker/detectron2/train_detectron2.py
else
    echo "[DOCKER] Running in CPU mode..."
    docker run -it \
        -v "$PROJECT_ROOT:/workspace" \
        -w /workspace \
        -e DETECTRON2_USE_CUDA=0 \
        detectron2-training \
        python docker/detectron2/train_detectron2.py
fi