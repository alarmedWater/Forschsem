FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Clone YOLACT
RUN git clone --depth 1 https://github.com/dbolya/yolact.git /workspace/yolact

# Install Python dependencies
RUN pip install --no-cache-dir \
    opencv-python-headless \
    matplotlib \
    scikit-image \
    pycocotools \
    pandas \
    pillow \
    tqdm

# Copy training script
COPY train_yolact.py /workspace/

# Set Python path
ENV PYTHONPATH=/workspace:/workspace/yolact