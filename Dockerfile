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

# Библиотеки CUDA и сборка sherpa-onnx с поддержкой GPU ставятся из локальной
# папки wheels/ (в git не входит): пять пакетов NVIDIA с PyPI и wheel sherpa-onnx
# для CUDA 12 / cuDNN 9. Как её наполнить — раздел «Подготовка к сборке образа»
# в README. Драйвер приходит с хоста через проброс GPU в контейнер.
COPY wheels/ /tmp/wheels/
RUN pip install --no-cache-dir --no-index --find-links /tmp/wheels \
    nvidia-cublas-cu12==12.4.5.8 nvidia-cudnn-cu12==9.1.0.70 \
    nvidia-nvjitlink-cu12==12.4.127 \
    && pip uninstall -y sherpa-onnx sherpa-onnx-core \
    && pip install --no-cache-dir --no-deps --no-index --find-links /tmp/wheels \
       nvidia-curand-cu12==10.3.5.147 nvidia-cuda-runtime-cu12==12.8.90 \
       'sherpa-onnx==1.13.8+cuda12.cudnn9.onnxruntime1.28.2' \
    && rm -rf /tmp/wheels
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.11/site-packages/nvidia/curand/lib:/usr/local/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:/usr/local/lib/python3.11/site-packages/nvidia/nvjitlink/lib

# Приложение
COPY config.py .
COPY transcriber.py .
COPY service.py .
COPY diarizer.py .
COPY ui.py .
COPY app.py .

# Директории (input — записи, output — расшифровки, whisper — кеш модели)
RUN mkdir -p /app/input /app/output /app/whisper
ENV INPUT_DIR=/app/input
ENV OUTPUT_DIR=/app/output

# Переменные окружения
ENV WHISPER_MODEL=large-v3-turbo
ENV WHISPER_DEVICE=auto
ENV WHISPER_BEAM_SIZE=5
ENV WHISPER_LANGUAGE=ru
ENV WHISPER_DOWNLOAD_ROOT=/app/whisper
ENV DIARIZATION_DIR=/app/whisper/diarization
ENV DIARIZATION_PROVIDER=cuda DIARIZATION_NUM_THREADS=2
# Время в meta.json и интерфейсе — по Москве
ENV TZ=Europe/Moscow
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7861

VOLUME ["/app/output"]

EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7861/')" || exit 1

CMD ["python", "app.py"]
