FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    git wget curl ffmpeg unzip vim build-essential \
    libgl1-mesa-glx libglib2.0-0 \
    libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip setuptools wheel --no-cache-dir

RUN pip install --no-cache-dir \
    torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir \
    antlr4-python3-runtime==4.13.2 \
    && pip install --no-cache-dir omegaconf==2.3.0 --no-deps \
    && pip install --no-cache-dir \
        numpy==1.23.5 scipy==1.8.1 scikit-learn==1.0.2 pandas==1.5.3 \
        matplotlib==3.7.1 tqdm==4.65.0 einops==0.6.0 timm==0.6.12 \
        transformers==4.27.4 datasets==2.10.1 \
        ffmpeg-python==0.2.0 webdataset==0.2.43 \
        h5py==3.9.0 regex==2022.7.9 Pillow==9.4.0 \
        wandb==0.17.8 future ftfy sentencepiece \
        gitpython pyarrow av==14.0.1 --prefer-binary

WORKDIR /workspace
WORKDIR /workspace/Synchformer

ENV PYTHONPATH=/workspace/Synchformer
ENV TORCH_CUDA_ARCH_LIST="8.6"

CMD ["/bin/bash"]
