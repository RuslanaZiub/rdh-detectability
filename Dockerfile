FROM python:3.12.3-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates unzip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY docker/lab/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt
# CPU-only PyTorch keeps the image smaller than the default CUDA-enabled wheel.
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
