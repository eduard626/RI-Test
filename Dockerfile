FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04


ARG CUDA_HOME=/usr/local/cuda-12.6
RUN export CUDA_HOME=${CUDA_HOME} 

ENV CUDA_HOME=${CUDA_HOME}
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# non GUI debian 
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget \
    curl \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    python-is-python3 \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

# Create and activate a virtual environment
ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

RUN python -m pip install --upgrade pip

# opencv and some tools
# we'll use Grounding DINO from GROUNDED-SAM2, so we replicate here
RUN python -m pip install opencv-python==4.10.0.82 \
    "setuptools>=62.3.0,<75.9"\
    wheel \
    numpy==1.26.4 \
    matplotlib \
    transformers \
    supervision \
    pycocotools \
    addict \
    yapf \
    timm

# torch and stuff
RUN python -m pip install --no-cache-dir \
    torch==2.7\
    torchvision==0.22.0\
    torchaudio

# get Grounded SAM2
RUN git clone https://github.com/IDEA-Research/Grounded-SAM-2.git

WORKDIR /Grounded-SAM-2

# get Grounding DINO checkpoints
RUN cd gdino_checkpoints && \
    bash download_ckpts.sh

# install
RUN python -m pip install --no-build-isolation -e grounding_dino

# copy the scripts to the workdir
COPY ./detection.py .
COPY ./detector.py .
COPY ./tracker.py .
COPY ./utils.py .

# copy the videos to the workdir
COPY ./*.mp4 ./

# copy the script
COPY ./run.sh .
RUN chmod +x run.sh

