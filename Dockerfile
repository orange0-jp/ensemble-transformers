FROM python:3.8.12-slim AS base
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES utility,compute
RUN apt-get update && apt-get install -y \
    ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libxrender-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
ARG PYTORCH="1.12.1"
ARG TORCHVISION="0.13.1"
ARG CUDA="113"

# Install python package.
WORKDIR /ensemble_transformers
COPY ./ /ensemble_transformers
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==${PYTORCH}+cu${CUDA} torchvision==${TORCHVISION}+cu${CUDA} -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install .

# Language settings
ENV LANG C.UTF-8
ENV LANGUAGE en_US

WORKDIR /workspace
