FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (needed for maturin build)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# uv might install to ~/.local/bin or ~/.cargo/bin depending on environment
ENV PATH="/root/.local/bin:/root/.cargo/bin:${PATH}"

WORKDIR /app

# Create virtual environment outside of /app so it persists if /app is bind-mounted
ENV VIRTUAL_ENV=/venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy project files to install dependencies
# We copy everything so we can build the rust extension
COPY . /app

# Install dependencies
# Explicitly install torch with CUDA 11.8 support first
RUN uv pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# Install the project (non-editable to keep compiled artifacts in venv)
# This builds the rust extension (maturin) and installs it to /venv/lib/python3.10/site-packages
RUN uv pip install .

# Set the default command to bash
CMD ["/bin/bash"]

