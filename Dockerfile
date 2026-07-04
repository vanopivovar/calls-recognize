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

# Приложение
COPY config.py .
COPY transcriber.py .
COPY ui.py .
COPY app.py .

# Директории (output — расшифровки, whisper — кеш модели)
RUN mkdir -p /app/output /app/whisper

# Переменные окружения
ENV WHISPER_MODEL=small
ENV WHISPER_COMPUTE_TYPE=int8
ENV WHISPER_LANGUAGE=ru
ENV WHISPER_DOWNLOAD_ROOT=/app/whisper
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7861

VOLUME ["/app/output"]

EXPOSE 7861

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7861/')" || exit 1

CMD ["python", "app.py"]
