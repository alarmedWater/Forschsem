FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

WORKDIR /workspace

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    build-essential \
    cmake \
    pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Upgrade pip toolchain early
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Python deps (versions compatible with torch 1.13 / cu116)
RUN pip install --no-cache-dir \
    "numpy==1.21.6" \
    cython \
    "pycocotools>=2.0.2" \
    opencv-python-headless \
    matplotlib \
    scikit-image \
    pandas \
    tqdm \
    pillow \
    seaborn

# Detectron2 prebuilt wheel for Torch 1.13 + CUDA 11.6
# (official wheels: https://dl.fbaipublicfiles.com/detectron2/wheels/cu116/torch1.13/index.html)
RUN pip install --no-cache-dir \
    'detectron2==0.6+cu116' \
    -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu116/torch1.13/index.html

# Copy your training script into the container
# (if your script lives elsewhere in the project, adjust the path or mount it with -v)
COPY train_detectron2.py /workspace/

# Environment
ENV PYTHONPATH=/workspace
# Optional: Some BLAS stacks like this setting; harmless if unused
ENV MKL_SERVICE_FORCE_INTEL=1

# Default command (you can override with `docker run ... python <path>` or via compose)
# CMD ["python", "/workspace/train_detectron2.py"]
