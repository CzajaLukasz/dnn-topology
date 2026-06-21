FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

# Instalacja zależności systemowych (dodano g++ dla pymetis)
RUN apt-get update && apt-get install -y \
    build-essential g++ cmake git wget libopenmpi-dev openmpi-bin tzdata \
    && rm -rf /var/lib/apt/lists/*

# Instalacja DIPHA do niezależnego folderu /opt/dipha
RUN git clone https://github.com/DIPHA/dipha.git /opt/dipha \
    && mkdir /opt/dipha/build \
    && cd /opt/dipha/build \
    && cmake .. \
    && make -j$(nproc)

ENV PATH="/opt/dipha/build:${PATH}"

# Instalacja bibliotek Pythona
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

ENV USER=root
ENV LOGNAME=root
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_cache
ENV TORCH_HOME=/tmp/torch_home
ENV MPLCONFIGDIR=/tmp/matplotlib_config