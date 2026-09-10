"""
Сервисный слой Calls Recognize — без Gradio.

Единый источник логики для интерфейса (ui.py) и MCP/HTTP-API.
Все функции возвращают простые JSON-совместимые словари/списки.
"""

import time
from pathlib import Path

import transcriber as T
from config import INPUT_DIR, OUTPUT_DIR


# ──────────────────────────────────────────────
# Модели
# ──────────────────────────────────────────────

def list_models(check_updates: bool = False) -> list[dict]:
    """
    Список всех моделей Whisper: скачана ли, размер, включена ли,
    есть ли обновление на Hugging Face (если check_updates=True — сетевой запрос).
    """
    return [T.model_info_row(n, check_remote=check_updates) for n in T.WHISPER_MODELS]


def download_model(name: str) -> dict:
    """Скачать модель (блокирующе) и загрузить в память. Возвращает {ok, message}."""
    ok, msg = T.ensure_model(name)
    return {"ok": ok, "message": msg, "model": name}


def delete_model(name: str) -> dict:
    """Удалить скачанную модель из кеша. Возвращает {ok, message}."""
    ok, msg = T.delete_model(name)
    return {"ok": ok, "message": msg, "model": name}


def set_enabled_models(names: list[str]) -> dict:
    """Задать, какие модели доступны для выбора и скачивания."""
    ok, msg = T.set_enabled_models(names)
    return {"ok": ok, "message": msg, "enabled": T.enabled_models()}


def enabled_models() -> list[str]:
    return T.enabled_models()


# ──────────────────────────────────────────────
# Расшифровка
# ──────────────────────────────────────────────

def iter_media_files(path: str | Path) -> list[Path]:
    return T.iter_media_files(path)


def transcribe_file(
    path: str,
    model: str | None = None,
    line_per_segment: bool = False,
    progress_callback=None,
) -> dict:
    """
    Расшифровать ОДИН файл → один результат (.txt + .srt + meta.json).
    progress_callback(fraction, desc) — необязательный колбэк прогресса.
    """
    p = Path(path)
    if not p.exists():
        return {"ok": False, "source": str(p), "error": "Файл не найден."}
    if not T.is_media_file(str(p)):
        return {"ok": False, "source": str(p), "error": "Это не видео/аудио."}

    name = (model or T.WHISPER_MODEL).strip()
    if name not in T.WHISPER_MODELS:
        return {"ok": False, "source": str(p), "error": f"Неизвестная модель: {name}"}

    if not T.is_model_cached(name):
        try:
            T.download_model_files(name)
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "source": str(p), "error": f"Не удалось скачать модель: {e}"}

    segments, info, meta = T.open_segments(str(p), name)
    if segments is None:
        return {"ok": False, "source": str(p), "error": str(meta)}

    duration = getattr(info, "duration", None)
    timed: list[tuple[float, float, str]] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            timed.append((seg.start, seg.end, text))
        if progress_callback and duration:
            try:
                frac = min((seg.end or 0.0) / duration, 0.99)
                progress_callback(frac, f"{seg.end:.0f}/{duration:.0f} сек")
            except Exception:
                pass

    if not timed:
        return {"ok": False, "source": str(p), "error": "Речь не распознана (тишина / нет дорожки)."}

    text = T.save_transcription(str(p), timed, line_per_segment)
    txt = T.transcript_path_for(p)
    srt = txt.with_suffix(".srt")
    meta_row = {
        "source": str(p),
        "model": name,
        "language": getattr(info, "language", None),
        "duration": round(duration, 1) if duration else None,
        "segments": len(timed),
        "line_per_segment": bool(line_per_segment),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    T.write_manifest(p, meta_row)
    return {"ok": True, "txt": str(txt), "srt": str(srt), "text": text, **meta_row}


def transcribe_path(
    path: str,
    model: str | None = None,
    line_per_segment: bool = False,
    progress_callback=None,
) -> list[dict]:
    """
    Расшифровать файл ИЛИ папку (рекурсивно). Один файл = один результат.
    Относительный путь ищется в INPUT_DIR.
    """
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = INPUT_DIR / p
    files = T.iter_media_files(p)
    if not files:
        return [{"ok": False, "source": str(p), "error": "Нет видео/аудио по этому пути."}]

    results = []
    n = len(files)
    for i, f in enumerate(files):
        def cb(frac, desc, _i=i):
            if progress_callback:
                progress_callback((_i + frac) / n, f"[{_i + 1}/{n}] {desc}")
        results.append(transcribe_file(str(f), model, line_per_segment, cb))
    return results


# ──────────────────────────────────────────────
# История
# ──────────────────────────────────────────────

def list_transcripts() -> list[dict]:
    """Все сохранённые расшифровки (свежие первыми) с метаданными."""
    out = []
    for txt in T.list_transcripts():
        meta = T.read_manifest(txt)
        out.append({
            "name": txt.parent.name,
            "txt": str(txt),
            "srt": str(txt.with_suffix(".srt")) if txt.with_suffix(".srt").exists() else None,
            "created": meta.get("created")
            or time.strftime("%Y-%m-%d %H:%M", time.localtime(txt.stat().st_mtime)),
            "model": meta.get("model"),
            "duration": meta.get("duration"),
            "segments": meta.get("segments"),
        })
    return out


def read_transcript(name_or_path: str) -> dict:
    """Текст и файлы расшифровки по имени папки или пути к .txt."""
    p = Path(name_or_path)
    if p.suffix != ".txt":
        p = OUTPUT_DIR / p.name / f"{p.name}.txt"
    text, files = T.read_transcript(p)
    return {"ok": p.exists(), "name": p.parent.name, "text": text, "files": files,
            "meta": T.read_manifest(p)}


# ──────────────────────────────────────────────
# API-обёртки для MCP (простые типы, без колбэков)
# ──────────────────────────────────────────────

def api_list_models(check_updates: bool = False) -> list[dict]:
    """Список моделей Whisper: downloaded, size_mb, enabled, update_available."""
    return list_models(check_updates=check_updates)


def api_download_model(name: str) -> dict:
    """Скачать модель Whisper по имени (например 'small', 'large-v3-turbo')."""
    return download_model(name)


def api_delete_model(name: str) -> dict:
    """Удалить скачанную модель из кеша по имени."""
    return delete_model(name)


def api_set_enabled_models(names: list[str]) -> dict:
    """Задать список доступных моделей (например ['small','medium'])."""
    return set_enabled_models(names)


def api_transcribe(path: str, model: str = "", line_per_segment: bool = False) -> list[dict]:
    """
    Расшифровать файл или папку. Путь — внутри контейнера (/app/input/...)
    или относительный к папке input. Один файл = один результат.
    """
    return transcribe_path(path, model or None, line_per_segment)


def api_list_transcripts() -> list[dict]:
    """Список сохранённых расшифровок с метаданными."""
    return list_transcripts()


def api_read_transcript(name: str) -> dict:
    """Текст расшифровки по имени (имя папки в output/) или пути к .txt."""
    return read_transcript(name)
