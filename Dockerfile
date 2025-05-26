FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04


ARG CUDA_HOME=/usr/local/cuda-12.6
RUN export CUDA_HOME=${CUDA_HOME} 
ENV CUDA_HOME=${CUDA_HOME}

# non GUI debian 
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget \
    curl \
    python-is-python3 \
    python3 \
    python3-pip \
    ffmpeg 

RUN python -m pip install --upgrade pip

# opencv and some tools
# we'll use Grounding DINO from GROUNDED-SAM2, so we replicate here
RUN pip install opencv-python==4.10.0.82 \
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
RUN pip install --no-cache-dir \
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
RUN pip install --no-build-isolation -e grounding_dino

# copy the scripts to the workdir
COPY ./detection.py .
COPY ./detector.py .
COPY ./tracker.py .
COPY ./utils.py .

# copy the videos to the workdir
COPY ./*.mp4 ./

# a temp directory for the output
RUN mkdir -p /Grounded-SAM-2/RI_OUTPUT
# set the entrypoint with the deafault video
ENTRYPOINT ["python", "detection.py", "--video", "AICandidateTest-FINAL.mp4", "--output", "RI_OUTPUT", "--visualize"]
