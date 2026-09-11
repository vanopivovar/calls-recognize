# ══════════════════════════════════════════════
# Calls Recognize — Docker Image
# Whisper ASR (faster-whisper) + Gradio + ffmpeg
# ══════════════════════════════════════════════

FROM python:3.11-slim AS base

LABEL maintainer="calls-recognize"
LABEL description="Call recording transcription with faster-whisper"

# ffmpeg — для декодирования видео/аудио дорожек
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Зависимости (faster-whisper тянет ctranslate2/av, БЕЗ torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CUDA-библиотеки для GPU-режима (CTranslate2 нужны cuBLAS и cuDNN 9).
# Wheels скачиваются на хосте в wheels/ (через прокси Docker Desktop большие
# файлы с PyPI приходят битыми) и ставятся из локальной папки:
#   py -m pip download nvidia-cublas-cu12==12.4.5.8 nvidia-cudnn-cu12==9.1.0.70 \
#      nvidia-nvjitlink-cu12==12.4.127 --only-binary=:all: --python-version 3.11 \
#      --implementation cp --platform manylinux2014_x86_64 -d wheels
# Драйвер приходит с хоста через --gpus all. Без GPU не мешают — откат на CPU.
COPY wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --no-index --find-links /tmp/wheels \
    nvidia-cublas-cu12==12.4.5.8 nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-nvjitlink-cu12==12.4.127 \
    && rm -rf /tmp/wheels
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib

# Приложение
COPY config.py .
COPY transcriber.py .
COPY service.py .
COPY ui.py .
COPY app.py .

# Директории (input — записи, output — расшифровки, whisper — кеш модели)
RUN mkdir -p /app/input /app/output /app/whisper
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

# Переменные окружения
ENV WHISPER_MODEL=small
ENV WHISPER_DEVICE=auto
ENV WHISPER_BEAM_SIZE=5
ENV WHISPER_LANGUAGE=ru
ENV WHISPER_DOWNLOAD_ROOT=/app/whisper
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7861

VOLUME ["/app/output"]

EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7861/')" || exit 1

CMD ["python", "app.py"]
