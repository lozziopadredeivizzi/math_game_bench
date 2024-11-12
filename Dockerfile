FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
LABEL maintainer="disi-unibo-nlp"

# Zero interaction (default answers to all questions)
ENV DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /

# Install general-purpose dependencies
RUN apt-get update -y && \
    apt-get install -y curl \
    git \
    bash \
    nano \
    wget \
    python3.10 \
    python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install wrapt --upgrade --ignore-installed
RUN pip install gdown

# Install PyTorch and related packages (part 1)
RUN pip install --upgrade torch

# Install other Python packages
RUN pip3 install --upgrade datasets
RUN pip3 install --upgrade wandb
RUN pip3 install --upgrade tokenizers
RUN pip3 install --upgrade tqdm
RUN pip3 install --upgrade nltk
RUN pip3 install --upgrade scipy
RUN pip3 install --upgrade huggingface_hub

RUN pip3 install transformers
RUN pip3 install peft

RUN pip3 install git+https://github.com/huggingface/accelerate.git
RUN pip3 install git+https://github.com/huggingface/trl.git

# required for flash attention
RUN pip3 install --upgrade packaging
RUN pip3 install --upgrade ninja
RUN MAX_JOBS=4 pip3 install --upgrade flash-attn --no-build-isolation
RUN pip3 install --upgrade bitsandbytes
RUN pip3 install --upgrade vllm

# Back to default frontend
ENV DEBIAN_FRONTEND=dialog