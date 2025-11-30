FROM nvcr.io/nvidia/cuda:12.6.0-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y \
    git \
    curl \
    make \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o miniforge.sh \
 && bash miniforge.sh -b -p "$CONDA_DIR" \
 && rm miniforge.sh

ENV PATH="$CONDA_DIR/bin:$PATH"

RUN conda install -y "ffmpeg>=4,<9" && conda clean -afy

USER ubuntu
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
USER root
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

WORKDIR /workspace