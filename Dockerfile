FROM python:3.7-slim

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget libopenmpi-dev openmpi-bin \
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
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app