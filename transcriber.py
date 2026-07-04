"""
Модуль расшифровки речи (ASR) из видео/аудио — для записей созвонов.

Использует faster-whisper (CTranslate2): быстро на CPU, хорошо понимает
русский, читает видеофайлы напрямую через встроенный ffmpeg/PyAV.

Модель загружается лениво при первом обращении и кешируется в памяти —
импорт этого модуля не тянет тяжёлых зависимостей и не качает модель.
"""

import os
import re
from pathlib import Path

from config import OUTPUT_DIR

# Размер модели Whisper. Чем больше — тем точнее и медленнее.
# tiny / base / small / medium / large-v3
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

# Доступные многоязычные модели с ориентиром по размеру скачивания (для UI).
# Английские (.en) и distil-модели не включены — для русского не подходят.
WHISPER_MODELS = {
    "tiny": "tiny (~75 МБ, быстрая, черновая)",
    "base": "base (~145 МБ)",
    "small": "small (~500 МБ, баланс)",
    "medium": "medium (~1.5 ГБ, точнее, медленнее)",
    "large-v1": "large-v1 (~3 ГБ, устаревшая)",
    "large-v2": "large-v2 (~3 ГБ, высокая точность)",
    "large-v3": "large-v3 (~3 ГБ, макс. точность)",
    "large-v3-turbo": "large-v3-turbo (~1.6 ГБ, почти как v3, но быстрее)",
}

# Тип вычислений CTranslate2 на CPU. int8 — быстро и экономно по памяти.
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

# Язык по умолчанию. Пусто/auto → автоопределение.
WHISPER_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "ru")

# Куда faster-whisper складывает скачанные веса.
WHISPER_DOWNLOAD_ROOT = os.environ.get("WHISPER_DOWNLOAD_ROOT") or None

# Форматы, которые считаем видео/аудио и отправляем на расшифровку.
MEDIA_EXTENSIONS = {
    # видео
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv",
    # аудио
    ".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac",
}

# Кеш загруженных моделей: {(имя, compute_type): WhisperModel}.
_models: dict = {}


def is_media_file(file_path: str) -> bool:
    """True, если расширение файла относится к видео/аудио."""
    return Path(file_path).suffix.lower() in MEDIA_EXTENSIONS


def transcript_path_for(source_path: str | Path) -> Path:
    """
    Куда сохраняется расшифровка для данного файла:
        output/<имя_видео>/<имя_видео>.txt
    """
    stem = Path(source_path).stem
    safe_stem = re.sub(r'[^\w\s.-]', '_', stem).strip() or "transcript"
    return OUTPUT_DIR / safe_stem / f"{safe_stem}.txt"


def _save_transcript(source_path: Path, text: str) -> Path:
    """Сохраняет распознанный текст и возвращает путь к .txt."""
    txt_path = transcript_path_for(source_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(text, encoding="utf-8")
    return txt_path


def _format_timestamp(seconds: float) -> str:
    """Секунды → таймкод SRT: ЧЧ:ММ:СС,мс"""
    ms = int(round(max(seconds, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(segments: list[tuple[float, float, str]]) -> str:
    """Формирует содержимое .srt из сегментов (начало, конец, текст)."""
    lines = []
    for i, (start, end, text) in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_timestamp(start)} --> {_format_timestamp(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def _save_srt(source_path: Path, segments: list[tuple[float, float, str]]) -> Path:
    """Сохраняет субтитры с таймкодами и возвращает путь к .srt."""
    srt_path = transcript_path_for(source_path).with_suffix(".srt")
    srt_path.parent.mkdir(parents=True, exist_ok=True)
    srt_path.write_text(_build_srt(segments), encoding="utf-8")
    return srt_path


def _get_model(model_name: str | None = None):
    """
    Лениво загружает (и кеширует) модель Whisper по имени.
    При первом обращении к незнакомой модели faster-whisper скачает её.
    """
    name = model_name or WHISPER_MODEL
    key = (name, WHISPER_COMPUTE_TYPE)
    if key in _models:
        return _models[key]

    from faster_whisper import WhisperModel

    model = WhisperModel(
        name,
        device="cpu",
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root=WHISPER_DOWNLOAD_ROOT,
    )
    _models[key] = model
    return model


# Файлы модели, которые тянет faster-whisper (те же allow_patterns).
_MODEL_FILES = [
    "config.json",
    "preprocessor_config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
]


def _repo_id(model_name: str) -> str:
    """Имя размера → репозиторий на Hugging Face (или сам id, если это путь)."""
    from faster_whisper.utils import _MODELS
    return _MODELS.get(model_name, model_name)


def is_model_cached(model_name: str) -> bool:
    """True, если модель уже скачана локально (в кеше), без обращения к сети."""
    import huggingface_hub
    try:
        huggingface_hub.snapshot_download(
            _repo_id(model_name),
            cache_dir=WHISPER_DOWNLOAD_ROOT,
            allow_patterns=_MODEL_FILES,
            local_files_only=True,
        )
        return True
    except Exception:
        return False


def download_model_files(model_name: str) -> None:
    """
    Скачивает файлы модели с Hugging Face с ВКЛЮЧЁННЫМ tqdm-прогрессом
    (в отличие от faster-whisper, который его глушит). Gradio перехватит
    этот прогресс и покажет проценты/МБ. Если уже скачано — быстрая проверка.
    """
    import huggingface_hub
    huggingface_hub.snapshot_download(
        _repo_id(model_name),
        cache_dir=WHISPER_DOWNLOAD_ROOT,
        allow_patterns=_MODEL_FILES,
    )


def ensure_model(model_name: str) -> tuple[bool, str]:
    """
    Скачивает (при необходимости) и загружает модель в память.
    Возвращает (успех, сообщение).
    """
    name = (model_name or WHISPER_MODEL).strip()
    if name not in WHISPER_MODELS:
        return False, f"[ERROR]Неизвестная модель: {name}"
    try:
        download_model_files(name)   # реальный прогресс через tqdm
        _get_model(name)             # загрузка в память (из кеша, быстро)
        return True, f"[OK]Модель «{name}» готова к работе ({WHISPER_COMPUTE_TYPE})."
    except ImportError:
        return False, "[ERROR]Не установлен faster-whisper (pip install faster-whisper)."
    except Exception as e:
        return False, f"[ERROR]Не удалось загрузить модель «{name}»: {str(e)}"


def transcribe_media(file_path: str, model_name: str | None = None) -> tuple[str | None, str]:
    """
    Расшифровывает речь из видео/аудио файла в текст выбранной моделью.

    Возвращает (text, debug_info).
    """
    debug = []
    path = Path(file_path)

    if not path.exists():
        return None, f"[ERROR]Файл не найден: {file_path}"

    name = (model_name or WHISPER_MODEL).strip()
    size_mb = path.stat().st_size / (1024 * 1024)
    debug.append(f"[INFO]Файл: {path.name} ({size_mb:.1f} MB)")
    debug.append(f"[INFO]Модель Whisper: {name} ({WHISPER_COMPUTE_TYPE})")

    try:
        model = _get_model(name)
    except ImportError:
        return None, (
            "[ERROR]Не установлен faster-whisper.\n"
            "Установите: pip install faster-whisper"
        )
    except Exception as e:
        return None, f"[ERROR]Не удалось загрузить модель Whisper: {str(e)}"

    try:
        language = WHISPER_LANGUAGE.strip().lower()
        if language in ("", "auto"):
            language = None

        segments, info = model.transcribe(
            str(path),
            language=language,
            beam_size=5,
            vad_filter=True,  # отсекаем тишину/паузы — точнее на созвонах
        )

        detected = getattr(info, "language", None)
        if detected:
            prob = getattr(info, "language_probability", 0.0) or 0.0
            debug.append(f"[INFO]Язык: {detected} ({prob:.0%})")
        duration = getattr(info, "duration", None)
        if duration:
            debug.append(f"[INFO]Длительность: {duration:.0f} сек ({duration/60:.1f} мин)")

        # segments — генератор: распознавание идёт по мере итерации.
        # Собираем тайминги для .srt вместе с текстом.
        timed_segments: list[tuple[float, float, str]] = []
        for seg in segments:
            seg_text = (seg.text or "").strip()
            if seg_text:
                timed_segments.append((seg.start, seg.end, seg_text))

        if not timed_segments:
            debug.append("[WARN]Речь не распознана (тишина или нет дорожки?)")
            return None, "\n".join(debug)

        text = " ".join(t for _, _, t in timed_segments)
        debug.append(f"[OK]Распознано фрагментов: {len(timed_segments)}")
        debug.append(f"[OK]Извлечено символов: {len(text)}")

        # Сохраняем расшифровку (.txt и .srt) в директорию с именем файла
        try:
            txt_path = _save_transcript(path, text)
            debug.append(f"[OK]Текст сохранён: {txt_path}")
        except Exception as e:
            debug.append(f"[WARN]Не удалось сохранить текст: {str(e)}")
        try:
            srt_path = _save_srt(path, timed_segments)
            debug.append(f"[OK]Субтитры сохранены: {srt_path}")
        except Exception as e:
            debug.append(f"[WARN]Не удалось сохранить субтитры: {str(e)}")

        return text, "\n".join(debug)

    except Exception as e:
        debug.append(f"[ERROR]Ошибка расшифровки: {str(e)}")
        return None, "\n".join(debug)
