FROM pytorch/pytorch:1.13.0-cuda11.6-cudnn8-devel

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install specific compatible versions
RUN pip install --no-cache-dir \
    "numpy==1.21.6" \
    "cython" \
    "pycocotools>=2.0.2" \
    opencv-python-headless \
    matplotlib \
    scikit-image \
    pandas \
    tqdm \
    pillow \
    seaborn

# Install Detectron2 with specific commit for compatibility
RUN pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git@v0.6'

# Alternative: If the above fails, try this instead:
# RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Copy the training script
COPY train_detectron2.py /workspace/

# Set environment variables
ENV PYTHONPATH=/workspace
ENV FORCE_CUDA="0"
ENV MKL_SERVICE_FORCE_INTEL=1