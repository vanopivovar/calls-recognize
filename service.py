"""
Сервисный слой Calls Recognize — без Gradio.

Единый источник логики для интерфейса (ui.py) и MCP/HTTP-API.
Все функции возвращают простые JSON-совместимые словари/списки.
"""

import json
import queue
import threading
import time
import uuid
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


# Активные расшифровки: job_id -> состояние (для «Стоп» с сохранением частичного).
_jobs: dict[str, dict] = {}


def _finalize(state: dict, partial: bool) -> dict:
    """Сохраняет .txt/.srt/meta.json из накопленных реплик и возвращает результат."""
    p = Path(state["path"])
    utterances = state.get("utterances") or [
        {"start": s["start"], "end": s["end"], "text": s["text"]} for s in state["segs"]
    ]
    txt = Path(state["txt"])
    text = T.write_outputs(txt, utterances, state["lps"])
    duration = state.get("duration")
    covered = round(max(u["end"] for u in utterances), 1) if utterances else 0.0
    speakers = sorted({u["speaker"] for u in utterances if u.get("speaker")})
    meta_row = {
        "source": str(p),
        "model": state["model"],
        "language": state.get("language"),
        "duration": round(duration, 1) if duration else None,
        "covered": covered,
        "segments": len(utterances),
        "partial": bool(partial),
        "line_per_segment": bool(state["lps"]),
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "recorded": T.recording_time(str(p)),
        "terms": len(state.get("terms") or []),
        "hallucinations_removed": state.get("removed", []),
        "lost_speech_repaired": state.get("repaired", []),
        "lost_speech_left": state.get("lost_left", []),
        "speakers": speakers,
        "diarization_error": state.get("diar_error"),
        "diarization_runtime": state.get("diar_runtime"),
        "repair_error": state.get("repair_error"),
        "failure": state.get("failure"),
    }
    mp = txt.parent / "meta.json"
    mp.write_text(json.dumps(meta_row, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "txt": str(txt), "srt": str(txt.with_suffix(".srt")), "text": text, **meta_row}


def _hms(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def iter_transcribe_file(
    path: str,
    model: str | None = None,
    line_per_segment: bool = False,
    job_id: str | None = None,
    diarize: bool = False,
    num_speakers: int = 0,
):
    """
    Генератор расшифровки ОДНОГО файла. Выдаёт:
      ("progress", fraction, desc)  — по ходу работы,
      ("error", message)            — при ошибке,
      ("done", result_dict)         — в конце (файлы сохранены).
    Этапы: распознавание → отсев галлюцинаций → повторное распознавание
    кусков с потерянной речью → (по запросу) разметка говорящих → сохранение.
    Цикл по сегментам живёт здесь, поэтому вызывающая сторона может
    прервать его (перестать итерироваться) — и Whisper остановится.
    Если задан job_id, состояние доступно stop_job() для частичного сохранения.
    """
    p = Path(path)
    if not p.exists():
        yield ("error", "Файл не найден."); return
    if not T.is_media_file(str(p)):
        yield ("error", "Это не видео/аудио."); return

    name = (model or T.WHISPER_MODEL).strip()
    if name not in T.WHISPER_MODELS:
        yield ("error", f"Неизвестная модель: {name}"); return

    if not T.is_model_cached(name):
        try:
            T.download_model_files(name)
        except Exception as e:  # noqa: BLE001
            yield ("error", f"Не удалось скачать модель: {e}"); return

    terms = T.get_terms()
    hotwords = T.hotwords_text(terms)
    segments, info, meta = T.open_segments(str(p), name, hotwords)
    if segments is None:
        yield ("error", str(meta)); return

    duration = getattr(info, "duration", None)
    state = {
        "path": str(p), "model": name, "lps": bool(line_per_segment),
        "txt": str(T.unique_output_txt(p)), "terms": terms,
        "segs": [], "removed": [], "duration": duration,
        "language": getattr(info, "language", None),
    }
    if job_id:
        _jobs[job_id] = state

    # Доля прогресса на распознавание; остальное — на повтор потерь и говорящих.
    asr_share = 0.75 if diarize else 0.9
    completed = False
    try:
        prev = None
        for seg in segments:
            d = T.segment_dict(seg)
            if d:
                if T.is_hallucination(d, prev):
                    state["removed"].append(f"{_hms(d['start'])} {d['text']}")
                else:
                    state["segs"].append(d)
                    prev = d
            if duration:
                frac = min((seg.end or 0.0) / duration, 0.99) * asr_share
                yield ("progress", frac, f"{seg.end:.0f}/{duration:.0f} сек")
        if not state["segs"]:
            completed = True
            yield ("error", "Речь не распознана (тишина / нет дорожки)."); return

        # Потерянная речь двух видов, оба куска перераспознаём отдельно, без VAD:
        #  • сегмент почти без слов (Whisper «проглотил» речь внутри окна);
        #  • пропуск между сегментами, где детектор речи слышит речь.
        yield ("progress", asr_share, "ищу потерянную речь")
        audio = None
        state["repaired"], state["lost_left"] = [], []
        # Сбой на этом этапе (например, кончилась видеопамять) не должен стоить
        # уже распознанного текста: записываем причину и идём сохранять.
        try:
            audio = T.load_audio(str(p))
            lost = [i for i, s in enumerate(state["segs"]) if T.is_lost_speech(s)]
            for k, i in enumerate(reversed(lost), 1):
                old = state["segs"][i]
                yield ("progress", asr_share, f"повтор потерянного куска {k}/{len(lost)} ({_hms(old['start'])})")
                new = T.retranscribe_range(name, audio, old["start"], old["end"], hotwords)
                old_words = len(old["text"].split())
                new_words = sum(len(s["text"].split()) for s in new)
                span = f"{_hms(old['start'])}–{_hms(old['end'])}"
                if new_words > old_words * 1.5 and new_words - old_words >= 5:
                    state["segs"][i:i + 1] = new
                    state["repaired"].append(f"{span}: было слов {old_words}, стало {new_words}")
                else:
                    state["lost_left"].append(f"{span}: {old['text']}")

            gaps = [g for g in T.find_gaps(state["segs"], duration)
                    if T.speech_share(audio, *g) >= T.GAP_MIN_SPEECH]
            for k, (a, b) in enumerate(gaps, 1):
                yield ("progress", asr_share, f"повтор пропуска {k}/{len(gaps)} ({_hms(a)})")
                new = T.retranscribe_range(name, audio, a, b, hotwords)
                new_words = sum(len(s["text"].split()) for s in new)
                span = f"{_hms(a)}–{_hms(b)}"
                if new_words >= 3:
                    state["segs"].extend(new)
                    state["repaired"].append(f"{span}: пропуск, восстановлено слов {new_words}")
                else:
                    state["lost_left"].append(f"{span}: пропуск, речь слышна, но не распознана")
        except Exception as e:  # noqa: BLE001
            state["repair_error"] = f"Поиск потерянной речи прерван: {e}"
        state["segs"].sort(key=lambda s: s["start"])
        state["repaired"].sort()
        state["lost_left"].sort()

        if diarize:
            try:
                import diarizer as D
                yield ("progress", asr_share, "разделяю по говорящим")
                if audio is None:
                    audio = T.load_audio(str(p))
                turns = D.diarize(audio, num_speakers)
                state["diar_runtime"] = D.runtime_info()
                state["utterances"] = D.assign_speakers(state["segs"], turns)
            except Exception as e:  # noqa: BLE001
                state["diar_error"] = f"Не удалось разделить по говорящим: {e}"

        result = _finalize(state, partial=False)
        completed = True
        yield ("done", result)
    except GeneratorExit:
        # Отмена: генератор брошен вызывающей стороной. Состояние оставляем —
        # его заберёт stop_job() и сохранит частичный результат.
        raise
    except Exception as e:  # noqa: BLE001
        # Сбой посреди работы: сохраняем то, что успело распознаться.
        completed = True
        state["failure"] = str(e)
        saved = None
        if state["segs"]:
            try:
                saved = _finalize(state, partial=True)
            except Exception:  # noqa: BLE001
                saved = None
        if saved:
            yield ("done", saved)
        else:
            yield ("error", f"Расшифровка прервана: {e}")
    finally:
        # При завершении и при сбое убираем состояние; при отмене оно остаётся.
        if completed and job_id:
            _jobs.pop(job_id, None)


def stop_job(job_id: str) -> dict | None:
    """
    Завершает остановленную расшифровку: сохраняет уже распознанное как
    частичный результат (partial=True). None — активной задачи нет.
    """
    state = _jobs.pop(job_id, None)
    if not state:
        return None
    if not state["segs"]:
        return {"ok": False, "source": state["path"], "error": "Ничего не успело распознаться."}
    return _finalize(state, partial=True)


def discard_job(job_id: str) -> None:
    """Забывает состояние расшифровки без сохранения (остаток от закрытой вкладки)."""
    _jobs.pop(job_id, None)


def transcribe_file(
    path: str,
    model: str | None = None,
    line_per_segment: bool = False,
    progress_callback=None,
    diarize: bool = False,
    num_speakers: int = 0,
) -> dict:
    """
    Расшифровать ОДИН файл → один результат (.txt + .srt + meta.json).
    Блокирующая обёртка над iter_transcribe_file (для MCP/API).
    """
    for ev in iter_transcribe_file(path, model, line_per_segment,
                                   diarize=diarize, num_speakers=num_speakers):
        if ev[0] == "progress":
            if progress_callback:
                try:
                    progress_callback(ev[1], ev[2])
                except Exception:
                    pass
        elif ev[0] == "error":
            return {"ok": False, "source": str(path), "error": ev[1]}
        elif ev[0] == "done":
            return ev[1]
    return {"ok": False, "source": str(path), "error": "Расшифровка прервана."}


OUTSIDE_INPUT = "Путь вне папки input: расшифровывать можно только файлы и папки из неё."


def _inside(p: Path, root: Path) -> Path | None:
    """Настоящий путь p, если он лежит внутри root (с учётом .. и ссылок), иначе None."""
    try:
        real = p.resolve()
        real.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return real


def _resolve(path: str) -> Path | None:
    """
    Путь для расшифровки через API. Относительный ищется в INPUT_DIR, абсолютный
    должен лежать в ней же. Всё, что вне INPUT_DIR, — None: сервис без пароля
    не должен читать произвольные файлы.
    """
    p = Path(str(path or "").strip())
    if not p.is_absolute():
        p = INPUT_DIR / p
    return _inside(p, INPUT_DIR)


def transcribe_path(
    path: str,
    model: str | None = None,
    line_per_segment: bool = False,
    progress_callback=None,
    diarize: bool = False,
    num_speakers: int = 0,
) -> list[dict]:
    """
    Расшифровать файл ИЛИ папку (рекурсивно). Один файл = один результат.
    Относительный путь ищется в INPUT_DIR.
    """
    p = _resolve(path)
    if p is None:
        return [{"ok": False, "source": str(path), "error": OUTSIDE_INPUT}]
    files = T.iter_media_files(p)
    if not files:
        return [{"ok": False, "source": str(p), "error": "Нет видео/аудио по этому пути."}]

    results = []
    n = len(files)
    for i, f in enumerate(files):
        def cb(frac, desc, _i=i):
            if progress_callback:
                progress_callback((_i + frac) / n, f"[{_i + 1}/{n}] {desc}")
        results.append(transcribe_file(str(f), model, line_per_segment, cb, diarize, num_speakers))
    return results


# ──────────────────────────────────────────────
# Фоновые задачи (для MCP/API: запуск не держит соединение)
# ──────────────────────────────────────────────

_bg_jobs: dict[str, dict] = {}
_bg_queue: "queue.Queue[str]" = queue.Queue()
_bg_lock = threading.Lock()
_bg_worker: threading.Thread | None = None


def _public(job: dict) -> dict:
    """Состояние задачи без полного текста (он большой — читать через read_transcript)."""
    res = [{k: v for k, v in r.items() if k != "text"} for r in job["results"]]
    return {k: v for k, v in job.items() if k not in ("results", "cancel", "opts")} | {"results": res}


def _run_bg(job: dict) -> None:
    files, n = job["files"], len(job["files"])
    job["status"] = "running"
    job["started"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    for i, f in enumerate(files):
        if job["cancel"]:
            job["results"].append({"ok": False, "source": f, "error": "Отменено до начала."})
            continue
        sub = f"{job['job_id']}:{i}"
        gen = iter_transcribe_file(f, job_id=sub, **job["opts"])
        result = None
        try:
            for ev in gen:
                if ev[0] == "progress":
                    job["progress"] = round((i + ev[1]) / n, 3)
                    job["desc"] = f"[{i + 1}/{n}] {Path(f).name}: {ev[2]}"
                elif ev[0] == "error":
                    result = {"ok": False, "source": f, "error": ev[1]}
                elif ev[0] == "done":
                    result = ev[1]
                if job["cancel"]:
                    break
        except Exception as e:  # noqa: BLE001
            result = {"ok": False, "source": f, "error": str(e)}
        finally:
            gen.close()
        if result is None:  # остановлено на середине — сохраняем распознанное
            result = stop_job(sub) or {"ok": False, "source": f, "error": "Остановлено."}
        job["results"].append(result)
    job["progress"] = 1.0
    job["status"] = "cancelled" if job["cancel"] else (
        "done" if any(r.get("ok") for r in job["results"]) else "error")
    job["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _worker_loop() -> None:
    # Задачи идут строго по одной: GPU одна, параллельные расшифровки только мешают.
    while True:
        jid = _bg_queue.get()
        job = _bg_jobs.get(jid)
        if job:
            try:
                _run_bg(job)
            except Exception as e:  # noqa: BLE001
                job["status"], job["error"] = "error", str(e)


def start_job(path: str, model: str | None = None, line_per_segment: bool = True,
              diarize: bool = False, num_speakers: int = 0) -> dict:
    """Ставит расшифровку файла или папки в очередь. Возвращает {job_id, status, files}."""
    global _bg_worker
    p = _resolve(path)
    if p is None:
        return {"ok": False, "error": OUTSIDE_INPUT}
    files = [str(f) for f in T.iter_media_files(p)]
    if not files:
        return {"ok": False, "error": f"Нет видео/аудио по пути: {p}"}
    jid = uuid.uuid4().hex[:8]
    _bg_jobs[jid] = {
        "ok": True, "job_id": jid, "status": "queued", "progress": 0.0, "desc": "в очереди",
        "files": files, "results": [], "cancel": False,
        "created": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "opts": {"model": model or None, "line_per_segment": line_per_segment,
                 "diarize": diarize, "num_speakers": int(num_speakers or 0)},
    }
    with _bg_lock:
        if _bg_worker is None or not _bg_worker.is_alive():
            _bg_worker = threading.Thread(target=_worker_loop, daemon=True, name="transcribe-jobs")
            _bg_worker.start()
    _bg_queue.put(jid)
    return _public(_bg_jobs[jid])


def job_status(job_id: str) -> dict:
    job = _bg_jobs.get(job_id)
    return _public(job) if job else {"ok": False, "error": f"Нет задачи {job_id}"}


def cancel_job(job_id: str) -> dict:
    """Остановить задачу: текущий файл сохранится частично, остальные пропустятся."""
    job = _bg_jobs.get(job_id)
    if not job:
        return {"ok": False, "error": f"Нет задачи {job_id}"}
    job["cancel"] = True
    return _public(job)


def list_jobs() -> list[dict]:
    return [_public(j) for j in sorted(_bg_jobs.values(), key=lambda j: j["created"], reverse=True)]


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
            "partial": bool(meta.get("partial")),
            "covered": meta.get("covered"),
            "speakers": meta.get("speakers") or [],
        })
    return out


def read_transcript(name_or_path: str) -> dict:
    """Текст и файлы расшифровки по имени папки или пути к .txt."""
    p = Path(str(name_or_path or "").strip())
    if p.suffix != ".txt":
        p = OUTPUT_DIR / p.name / f"{p.name}.txt"
    elif not p.is_absolute() and not p.exists():
        p = OUTPUT_DIR / p
    real = _inside(p, OUTPUT_DIR)
    if real is None:  # читать можно только расшифровки из папки результатов
        return {"ok": False, "name": "", "text": "", "files": [], "meta": {},
                "error": "Путь вне папки output."}
    p = real
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


def api_transcribe(path: str, model: str = "", line_per_segment: bool = False,
                   diarize: bool = False, num_speakers: int = 0) -> list[dict]:
    """
    Расшифровать файл или папку из папки input. Путь — относительный к ней
    или полный внутри контейнера (/app/input/...). Один файл = один результат.
    """
    return transcribe_path(path, model or None, line_per_segment,
                           diarize=diarize, num_speakers=num_speakers)


def api_start_transcribe(path: str, model: str = "", line_per_segment: bool = True,
                         diarize: bool = False, num_speakers: int = 0) -> dict:
    return start_job(path, model or None, line_per_segment, diarize, num_speakers)


def api_get_terms() -> dict:
    return {"ok": True, "terms": T.get_terms()}


def api_set_terms(terms: list[str]) -> dict:
    ok, msg = T.set_terms(terms)
    return {"ok": ok, "message": msg, "terms": T.get_terms()}


def api_list_transcripts() -> list[dict]:
    """Список сохранённых расшифровок с метаданными."""
    return list_transcripts()


def api_read_transcript(name: str) -> dict:
    """Текст расшифровки по имени (имя папки в output/) или пути к .txt."""
    return read_transcript(name)


def api_runtime() -> dict:
    """Среда выполнения: device (cuda/cpu), compute_type, gpu_name, cpu_threads, beam_size."""
    import diarizer as D
    return {**T.runtime_info(), "diarization": D.runtime_info()}
