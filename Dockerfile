FROM python:3.11-slim

ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
# Avoiding user interaction with libopencv-dev
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && apt install -y \
    git tmux
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    vim \
    libopencv-dev \
    wget \
    zsh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ARG PYTORCH="2.0.1"
ARG TORCHVISION="0.15.2"
ARG CUDA="118"

# Zsh install
ENV SHELL /bin/zsh
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t robbyrussell

# Install python package.
COPY ./ /tmp/ensemble_transformers
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==${PYTORCH}+cu${CUDA} torchvision==${TORCHVISION}+cu${CUDA} --extra-index-url https://download.pytorch.org/whl/cu${CUDA} && \
    pip install /tmp/ensemble_transformers

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US
