# Use the official NVIDIA CUDA base image
FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        software-properties-common \
        curl \
        git \
        ffmpeg \
        gosu \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.13 \
        python3.13-venv \
        python3.13-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.13 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY download_data.py .

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]