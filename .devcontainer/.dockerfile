FROM nvcr.io/nvidia/pytorch:25.10-py3

RUN conda install -y "ffmpeg<8" -c conda-forge && conda clean -afy

RUN pip install uv && \
    pip install torchcodec --index-url=https://download.pytorch.org/whl/cu126

WORKDIR /workspace
