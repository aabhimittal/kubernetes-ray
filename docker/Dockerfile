# Multi-stage build for the Ray + Kubernetes GPU training image.
# Base: NVIDIA CUDA runtime so torch can use the GPU inside the container.
# The NVIDIA GPU Operator provides the host driver; we only need the runtime.
ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3.10-venv \
        git \
        curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY docker/requirements.txt /app/docker/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /app/docker/requirements.txt

# Copy the project and install it.
COPY . /app
RUN pip install -e .

# Ray dashboard (8265), app dashboard/metrics (8266), Ray client (10001).
EXPOSE 8265 8266 10001

# Default: start a Ray head node. Workers override the command via KubeRay.
CMD ["ray", "start", "--head", "--dashboard-host=0.0.0.0", "--block"]
