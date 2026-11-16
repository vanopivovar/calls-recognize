"""
Разделение записи по говорящим (диаризация) — sherpa-onnx, без torch.

Две ONNX-модели, качаются один раз при первом использовании в DIARIZATION_DIR:
  • сегментация pyannote 3.0 — где в записи говорит кто-то один, где несколько;
  • эмбеддинги голоса NeMo TitaNet-large — чей это голос (WeSpeaker ResNet34 на
    созвонах не различал участников, слышных через один канал).
Дальше реплики кластеризуются: число говорящих задаётся или подбирается по порогу.

Имена сервис не знает — говорящие нумеруются по порядку первого появления.
"""

import os
import ctypes
import json
import subprocess
import sys
import tempfile
from importlib.metadata import version
import tarfile
import urllib.request
from bisect import bisect_right
from pathlib import Path

DIARIZATION_DIR = Path(os.environ.get(
    "DIARIZATION_DIR",
    str(Path(os.environ.get("WHISPER_DOWNLOAD_ROOT") or "whisper") / "diarization"),
))

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_URL = os.environ.get(
    "DIARIZATION_SEGMENTATION_URL",
    f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
)
EMBEDDING_URL = os.environ.get(
    "DIARIZATION_EMBEDDING_URL",
    f"{_RELEASES}/speaker-recongition-models/nemo_en_titanet_large.onnx",
)

# Порог кластеризации при автоподборе числа говорящих: больше — меньше говорящих.
DIARIZATION_THRESHOLD = float(os.environ.get("DIARIZATION_THRESHOLD", "0.5") or 0.5)
DIARIZATION_PROVIDER = os.environ.get("DIARIZATION_PROVIDER", "cuda").strip().lower()
DIARIZATION_NUM_THREADS = int(os.environ.get("DIARIZATION_NUM_THREADS", "2"))
if DIARIZATION_PROVIDER not in ("cuda", "cpu"):
    raise ValueError("DIARIZATION_PROVIDER must be cuda or cpu")
if not 1 <= DIARIZATION_NUM_THREADS <= 12:
    raise ValueError("DIARIZATION_NUM_THREADS must be between 1 and 12")

# Второй проход — склейка кластеров по усреднённому отпечатку голоса.
# Сходство отпечатков (косинус) не ниже порога — это один человек. 0 — выкл.
DIARIZATION_MERGE_THRESHOLD = float(os.environ.get("DIARIZATION_MERGE_THRESHOLD", "0.6") or 0)
MERGE_SMALL_THRESHOLD = 0.4    # мелкий остаток вливается в крупный кластер при таком сходстве
MERGE_SMALL_SECONDS = 15.0     # «мелкий» — меньше стольких секунд речи на всю запись
MERGE_SAMPLE_SECONDS = 60.0    # сколько речи кластера идёт в отпечаток
MERGE_MIN_VOICE = 1.5          # меньше — отпечаток ненадёжен, кластер отдаётся соседу
MERGE_CRUMB_SECONDS = 5.0      # голос, которому после склейки досталось меньше, — обрывок, а не участник
SMOOTH_MAX_WORDS = 2           # обрывок не длиннее стольких слов...
SMOOTH_MAX_SECONDS = 1.0       # ...и короче стольких секунд внутри чужой фразы — не реплика

SPEAKER_LABEL = "Говорящий {n}"

_engine = None
_engine_clusters = None
_last_worker_runtime = None


def runtime_info() -> dict:
    """Configured device, not a claim that an uninitialized engine was exercised."""
    return {"provider": DIARIZATION_PROVIDER, "cpu_threads": DIARIZATION_NUM_THREADS,
            "engine_initialized": _engine is not None, "threshold": DIARIZATION_THRESHOLD,
            "merge_threshold": DIARIZATION_MERGE_THRESHOLD,
            "sherpa_version": version("sherpa-onnx"), "isolated_process": True,
            "last_execution_succeeded": _last_worker_runtime is not None}


def _require_cuda() -> None:
    # A CPU wheel may silently ignore the requested CUDA provider. Fail visibly instead.
    if "+cuda" not in version("sherpa-onnx"):
        raise RuntimeError("CUDA diarization requires the pinned sherpa-onnx CUDA wheel")
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        if driver.cuInit(0) != 0:
            raise RuntimeError("NVIDIA driver initialization failed")
    except OSError as exc:
        raise RuntimeError("NVIDIA GPU is unavailable for diarization; no CPU fallback") from exc


def _download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dest)


def ensure_models() -> tuple[Path, Path]:
    """Скачивает модели при первом обращении. Возвращает (сегментация, эмбеддинги)."""
    DIARIZATION_DIR.mkdir(parents=True, exist_ok=True)
    seg_dir = DIARIZATION_DIR / "sherpa-onnx-pyannote-segmentation-3-0"
    seg_model = seg_dir / "model.onnx"
    if not seg_model.exists():
        archive = DIARIZATION_DIR / "segmentation.tar.bz2"
        _download(SEGMENTATION_URL, archive)
        with tarfile.open(archive, "r:bz2") as t:
            t.extractall(DIARIZATION_DIR, filter="data")
        archive.unlink(missing_ok=True)
    emb_model = DIARIZATION_DIR / Path(EMBEDDING_URL).name
    if not emb_model.exists():
        _download(EMBEDDING_URL, emb_model)
    return seg_model, emb_model


def _get_engine(num_speakers: int):
    """Движок sherpa-onnx (пересоздаётся, если поменялось число говорящих)."""
    global _engine, _engine_clusters
    clusters = num_speakers if num_speakers and num_speakers > 0 else -1
    if _engine is not None and _engine_clusters == clusters:
        return _engine
    import sherpa_onnx

    seg_model, emb_model = ensure_models()
    threads = DIARIZATION_NUM_THREADS
    if DIARIZATION_PROVIDER == "cuda":
        _require_cuda()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model)),
            num_threads=threads,
            provider=DIARIZATION_PROVIDER,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_model), num_threads=threads, provider=DIARIZATION_PROVIDER),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=clusters, threshold=DIARIZATION_THRESHOLD),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Неверная конфигурация диаризации (проверьте файлы моделей)")
    _engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    _engine_clusters = clusters
    return _engine


def _diarize_in_process(audio, num_speakers: int = 0, progress_callback=None) -> list[tuple[float, float, int]]:
    """
    audio — float32 16 кГц моно. Возвращает отрезки (начало, конец, номер говорящего),
    отсортированные по времени. progress_callback(fraction) — доля обработанного.
    """
    sd = _get_engine(num_speakers)

    def cb(done: int, total: int) -> int:
        if progress_callback and total:
            try:
                progress_callback(done / total)
            except Exception:
                pass
        return 0

    result = sd.process(audio, callback=cb).sort_by_start_time()
    return [(r.start, r.end, r.speaker) for r in result]


def _cluster_voiceprints(audio, turns) -> tuple[dict[int, "np.ndarray"], dict[int, float]]:
    """Отпечаток голоса каждого кластера по его самым длинным отрезкам
    (до MERGE_SAMPLE_SECONDS суммарно) и суммарная длительность кластеров.
    Кластеры, где речи меньше MERGE_MIN_VOICE секунд, отпечатка не получают."""
    import numpy as np
    import sherpa_onnx

    _, emb_model = ensure_models()
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(emb_model), num_threads=DIARIZATION_NUM_THREADS, provider=DIARIZATION_PROVIDER))
    by: dict[int, list[tuple[float, float]]] = {}
    for s, e, k in turns:
        by.setdefault(k, []).append((s, e))
    durations = {k: sum(e - s for s, e in v) for k, v in by.items()}
    prints = {}
    for k, spans in by.items():
        if durations[k] < MERGE_MIN_VOICE:
            continue
        parts, total = [], 0.0
        for s, e in sorted(spans, key=lambda t: t[0] - t[1]):
            if total >= MERGE_SAMPLE_SECONDS:
                break
            parts.append(np.asarray(audio[int(s * 16000):int(e * 16000)], dtype=np.float32))
            total += e - s
        stream = extractor.create_stream()
        stream.accept_waveform(16000, np.concatenate(parts))
        stream.input_finished()
        v = np.array(extractor.compute(stream), dtype=np.float32)
        prints[k] = v / (np.linalg.norm(v) + 1e-9)
    return prints, durations


def merge_plan(prints: dict, durations: dict[int, float]) -> dict[int, int]:
    """Какой кластер в какой вливается: {исходный номер: итоговый номер}.

    1. Пока есть пара кластеров с похожими отпечатками (сходство не ниже
       DIARIZATION_MERGE_THRESHOLD), самая похожая пара склеивается; отпечаток
       общего кластера — среднее с весом по длительности.
    2. Мелкий остаток (меньше MERGE_SMALL_SECONDS речи) вливается в самый
       похожий крупный кластер, если сходство не ниже MERGE_SMALL_THRESHOLD.
    Кластеры без отпечатка здесь не трогаются."""
    import numpy as np

    vec = {k: v.copy() for k, v in prints.items()}
    dur = {k: durations[k] for k in prints}
    target = {k: k for k in durations}

    def absorb(a: int, b: int) -> None:  # b вливается в a
        v = vec[a] * dur[a] + vec[b] * dur[b]
        vec[a] = v / (np.linalg.norm(v) + 1e-9)
        dur[a] += dur[b]
        del vec[b], dur[b]
        for k, t in target.items():
            if t == b:
                target[k] = a

    while len(vec) > 1:
        keys = list(vec)
        best, pair = DIARIZATION_MERGE_THRESHOLD, None
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                sim = float(vec[a] @ vec[b])
                if sim >= best:
                    best, pair = sim, (a, b)
        if not pair:
            break
        a, b = pair if dur[pair[0]] >= dur[pair[1]] else pair[::-1]
        absorb(a, b)

    for b in sorted((k for k in vec if dur[k] < MERGE_SMALL_SECONDS), key=lambda k: dur[k]):
        big = [a for a in vec if a != b and dur[a] >= MERGE_SMALL_SECONDS]
        if not big:
            break
        a = max(big, key=lambda a: float(vec[a] @ vec[b]))
        if float(vec[a] @ vec[b]) >= MERGE_SMALL_THRESHOLD:
            absorb(a, b)
    return target


def merge_similar_speakers(audio, turns: list[tuple[float, float, int]]) -> list[tuple[float, float, int]]:
    """Второй проход: первичная кластеризация sherpa работает по коротким кускам
    и на длинной записи дробит один голос на десятки кластеров. Здесь кластеры
    склеиваются по усреднённому отпечатку голоса (он устойчивее), обрывки без
    отпечатка отдаются соседнему по времени говорящему."""
    if DIARIZATION_MERGE_THRESHOLD <= 0 or len({k for _, _, k in turns}) < 2:
        return turns
    prints, durations = _cluster_voiceprints(audio, turns)
    target = merge_plan(prints, durations)
    out = [(s, e, target[k]) for s, e, k in turns]
    # Обрывки: голос, которому на всю запись досталось меньше MERGE_CRUMB_SECONDS
    # (в том числе кластеры без отпечатка), отдаётся ближайшему по времени участнику.
    total: dict[int, float] = {}
    for s, e, k in out:
        total[k] = total.get(k, 0.0) + e - s
    crumbs = {k for k, d in total.items() if d < MERGE_CRUMB_SECONDS}
    solid = [i for i, (_, _, k) in enumerate(out) if k not in crumbs]
    if solid and crumbs:
        from bisect import bisect_left
        for i, (s, e, k) in enumerate(out):
            if k not in crumbs:
                continue
            j = bisect_left(solid, i)
            near = [solid[x] for x in (j - 1, j) if 0 <= x < len(solid)]
            n = min(near, key=lambda x: min(abs(out[x][0] - e), abs(s - out[x][1])))
            out[i] = (s, e, out[n][2])
    return out


def diarize(audio, num_speakers: int = 0, progress_callback=None) -> list[tuple[float, float, int]]:
    """Isolate sherpa CUDA/ORT from Whisper/VAD native libraries in the parent.

    The worker memory-maps a temporary float32 array, avoiding a second full copy
    of long recordings. Neither the source recording nor transcript is modified.
    """
    global _last_worker_runtime
    import numpy as np
    _last_worker_runtime = None
    with tempfile.TemporaryDirectory(prefix="calls-diarization-") as folder:
        data = Path(folder) / "audio.npy"
        result = Path(folder) / "turns.json"
        np.save(data, audio, allow_pickle=False)
        if progress_callback:
            progress_callback(0.0)
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(data),
             str(result), str(int(num_speakers or 0))],
            capture_output=True, text=True, timeout=7200,
        )
        if proc.returncode:
            raise RuntimeError("Diarization worker failed (no CPU fallback): "
                               + (proc.stderr or str(proc.returncode))[-1500:])
        payload = json.loads(result.read_text(encoding="utf-8"))
        if payload["runtime"]["provider"] != DIARIZATION_PROVIDER:
            raise RuntimeError("Diarization worker used an unexpected provider")
        _last_worker_runtime = payload["runtime"]
        if progress_callback:
            progress_callback(1.0)
        return [tuple(t) for t in payload["turns"]]


def _speaker_at(turns: list[tuple[float, float, int]], starts: list[float], t: float) -> int | None:
    """Говорящий в момент t: отрезок, который его накрывает, иначе ближайший."""
    if not turns:
        return None
    i = bisect_right(starts, t) - 1
    best, best_dist = None, None
    for j in (i, i + 1):
        if 0 <= j < len(turns):
            s, e, spk = turns[j]
            dist = 0.0 if s <= t <= e else min(abs(t - s), abs(t - e))
            if best_dist is None or dist < best_dist:
                best, best_dist = spk, dist
    return best


def assign_speakers(segments: list[dict], turns: list[tuple[float, float, int]]) -> list[dict]:
    """
    Делит сегменты Whisper на реплики по говорящим. Говорящий определяется для
    каждого слова (по середине слова), подряд идущие слова одного говорящего
    склеиваются. Возвращает [{start, end, text, speaker}].
    """
    starts = [t[0] for t in turns]
    order: dict[int, str] = {}

    def label(spk: int | None) -> str | None:
        if spk is None:
            return None
        if spk not in order:
            order[spk] = SPEAKER_LABEL.format(n=len(order) + 1)
        return order[spk]

    out: list[dict] = []
    for seg in segments:
        words = seg.get("words") or []
        if not words:
            mid = (seg["start"] + seg["end"]) / 2
            out.append({**_strip(seg), "speaker": label(_speaker_at(turns, starts, mid))})
            continue
        # Сначала говорящий каждого слова, потом сглаживание, и только потом
        # номера: иначе голос, от которого после сглаживания не осталось реплик,
        # всё равно занял бы номер.
        marks = [[ws, we, w, _speaker_at(turns, starts, (ws + we) / 2)] for ws, we, w in words]
        runs: list[list[int]] = []          # [первое слово, последнее слово] одного говорящего
        for i, m in enumerate(marks):
            if runs and marks[runs[-1][0]][3] == m[3]:
                runs[-1][1] = i
            else:
                runs.append([i, i])
        # Обрывок в одно-два слова короче секунды посреди фразы одного и того же
        # говорящего — ошибка границы отрезков, а не реплика другого человека.
        for j in range(1, len(runs) - 1):
            a, b = runs[j]
            before, after = marks[runs[j - 1][0]][3], marks[runs[j + 1][0]][3]
            if before == after and b - a < SMOOTH_MAX_WORDS and marks[b][1] - marks[a][0] < SMOOTH_MAX_SECONDS:
                for i in range(a, b + 1):
                    marks[i][3] = before
        run: dict | None = None
        for ws, we, w, raw in marks:
            spk = label(raw)
            if run and run["speaker"] == spk:
                run["end"] = we
                run["text"] += w
            else:
                if run:
                    out.append(run)
                run = {"start": ws, "end": we, "text": w, "speaker": spk}
        if run:
            out.append(run)
    for u in out:
        u["text"] = u["text"].strip()
    return [u for u in out if u["text"]]


def _strip(seg: dict) -> dict:
    return {"start": seg["start"], "end": seg["end"], "text": seg["text"]}


if __name__ == "__main__":
    import numpy as np
    if len(sys.argv) != 5 or sys.argv[1] != "--worker":
        raise SystemExit("Internal diarization worker: --worker audio.npy turns.json speakers")
    samples = np.load(sys.argv[2], mmap_mode="r", allow_pickle=False)
    turns = _diarize_in_process(samples, int(sys.argv[4]))
    if int(sys.argv[4]) <= 0:   # число говорящих задано вручную — не трогаем
        turns = merge_similar_speakers(samples, turns)
    Path(sys.argv[3]).write_text(
        json.dumps({"turns": turns, "runtime": runtime_info()}), encoding="utf-8")
