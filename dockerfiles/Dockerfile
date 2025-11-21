# --- Stage 1: Build Environment (uv + PyTorch) ---
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS builder

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-venv \
        python3.12-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv (official recommended method)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Working directory
WORKDIR /workspace

# Create Python virtual environment with uv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
RUN uv venv /opt/venv --python python3.12
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch first (CUDA 12.8 compatible)
RUN uv pip install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy entire project files (including README.md)
COPY . .

# Install project dependencies and the project itself
# UV_PROJECT_ENVIRONMENT is set, so installation goes to /opt/venv
RUN uv sync --no-dev


# --- Stage 2: Runtime Environment (Final Image) ---
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Explicitly set CUDA library paths in environment variables
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install packages required for runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        git \
        graphviz \
        htop \
        nvtop \
        tree \
        vim \
        curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.12 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy uv command and completed virtual environment from build stage
COPY --from=builder /bin/uv /bin/uvx /bin/
COPY --from=builder /opt/venv /opt/venv

# Activate virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Working directory
WORKDIR /workspace

# Default command not set, specified in docker-compose.yaml
