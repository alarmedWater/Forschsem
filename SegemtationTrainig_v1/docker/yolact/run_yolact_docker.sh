#!/bin/bash
# run_yolact_docker.sh

set -euo pipefail

MODE="${1:-cpu}"  # cpu (sanity) | gpu (full)

DOCKER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DOCKER_DIR/../.." && pwd)"

echo "[YOLACT] Mode: $MODE"

if [[ "$MODE" == "cpu" ]]; then
    echo "🚀 CPU SANITY TEST - Schneller Test (5-15 Minuten)"
    echo "   - 50 Bilder"
    echo "   - 500 Iterationen" 
    echo "   - Validiert Daten/Config"
    
    cd "$DOCKER_DIR"
    docker build -t yolact-cpu -f Dockerfile.cpu .  # Separate CPU Dockerfile
    
    docker run -it \
        -v "$PROJECT_ROOT:/workspace" \
        -w /workspace \
        -e YOLACT_USE_CUDA=0 \
        yolact-cpu \
        python docker/yolact/train_yolact.py

elif [[ "$MODE" == "gpu" ]]; then
    echo "🏎️  GPU FULL TRAINING - Volles Training (mehrere Stunden)"
    echo "   - Komplettes Dataset"
    echo "   - 80,000 Iterationen"
    
    cd "$DOCKER_DIR" 
    docker build -t yolact-gpu -f Dockerfile.gpu .  # GPU Dockerfile
    
    docker run --gpus all -it \
        -v "$PROJECT_ROOT:/workspace" \
        -w /workspace \
        -e YOLACT_USE_CUDA=1 \
        yolact-gpu \
        python docker/yolact/train_yolact.py

elif [[ "$MODE" == "test" ]]; then
    echo "🧪 EXTREME SANITY - Superschnell (1-2 Minuten)"
    echo "   - 10 Bilder"
    echo "   - 100 Iterationen"
    echo "   - Nur grundlegende Funktionalität"
    
    # Setze extreme CPU-Parameter
    docker run -it \
        -v "$PROJECT_ROOT:/workspace" \
        -w /workspace \
        -e YOLACT_USE_CUDA=0 \
        -e YOLACT_QUICK_TEST=1 \
        yolact-cpu \
        python docker/yolact/train_yolact.py
fi