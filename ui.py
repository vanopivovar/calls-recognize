"""
Gradio-интерфейс Calls Recognize — расшифровка созвонов
"""

import gradio as gr
from pathlib import Path

from transcriber import (
    is_media_file,
    transcribe_media,
    transcript_path_for,
    ensure_model,
    is_model_cached,
    model_status_text,
    download_model_files,
    model_total_bytes,
    downloaded_bytes,
    list_transcripts,
    read_transcript,
    WHISPER_MODEL,
    WHISPER_MODELS,
)


CUSTOM_CSS = """
.gradio-container {
    max-width: 1000px !important;
    margin: auto !important;
    background: #1a1d24 !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.header-text {
    text-align: center;
    margin-bottom: 2rem;
    padding: 2rem 1rem;
    background: linear-gradient(135deg, #252a33 0%, #2d3440 100%);
    border-radius: 12px;
    border: 1px solid #353b47;
}
.header-text h1 {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
    color: #e4e6eb;
    font-weight: 600;
}
.header-text p {
    color: #b0b8c1;
    font-size: 0.95rem;
    line-height: 1.6;
}
button {
    background: #4a6785 !important;
    color: #e4e6eb !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}
button:hover {
    background: #5b7c99 !important;
    transform: translateY(-1px);
}
input, textarea, select {
    background: #2d3440 !important;
    color: #e4e6eb !important;
    border: 1px solid #353b47 !important;
    border-radius: 8px !important;
}
label {
    color: #b0b8c1 !important;
    font-weight: 500 !important;
}
"""


def _build_theme() -> gr.themes.Base:
    """Тёмная тема приложения."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#1a1d24",
        body_background_fill_dark="#1a1d24",
        block_background_fill="#252a33",
        block_background_fill_dark="#252a33",
        input_background_fill="#2d3440",
        input_background_fill_dark="#2d3440",
        button_primary_background_fill="#4a6785",
        button_primary_background_fill_hover="#5b7c99",
        button_primary_text_color="#e4e6eb",
    )


def _model_key(choice: str) -> str:
    """Из подписи в дропдауне достаёт ключ модели (первое слово)."""
    return (choice or WHISPER_MODEL).split()[0]


def model_status_wrapper(model_choice: str) -> str:
    """Статус выбранной модели (скачана / нет) — для дропдауна и загрузки страницы."""
    return model_status_text(_model_key(model_choice))


def _download_gen(name: str, progress):
    """
    Общий генератор скачивания модели с РЕАЛЬНЫМ прогрессом:
    качает в фоновом потоке и раз в секунду опрашивает размер на диске.
    Выдаёт строки статуса; двигает полосу прогресса.
    """
    import threading
    import time

    if is_model_cached(name):
        yield f"✅ Модель «{name}» уже скачана — загружаю в память..."
        ensure_model(name)
        yield model_status_text(name)
        return

    total = model_total_bytes(name)
    err = {}

    def _run():
        try:
            download_model_files(name)
        except Exception as e:  # noqa: BLE001
            err["e"] = e

    th = threading.Thread(target=_run, daemon=True)
    th.start()

    mb = 1024 * 1024
    while th.is_alive():
        done = downloaded_bytes(name)
        if total:
            frac = min(done / total, 0.99)
            try:
                progress(frac, desc=f"{done/mb:.0f}/{total/mb:.0f} МБ")
            except Exception:
                pass
            yield f"⏳ Скачиваю «{name}»: {done/mb:.0f} / {total/mb:.0f} МБ ({frac*100:.0f}%)"
        else:
            yield f"⏳ Скачиваю «{name}»: {done/mb:.0f} МБ скачано..."
        time.sleep(1.0)

    th.join()
    if err:
        yield f"❌ Не удалось скачать «{name}»: {err['e']}"
        return

    ensure_model(name)  # загрузка в память из кеша
    yield model_status_text(name)


def download_model_wrapper(model_choice: str, progress=gr.Progress()):
    """Скачивает/загружает выбранную модель заранее (без расшифровки)."""
    name = _model_key(model_choice)
    yield from _download_gen(name, progress)


def transcribe_wrapper(media_file, model_choice: str, progress=gr.Progress()):
    """
    Расшифровка видео/аудио в текст выбранной моделью (генератор).
    Показывает статус и реальный прогресс: скачивание модели (если нужно) +
    распознавание по сегментам. Выдаёт: (статус, текст, [файлы .txt/.srt]).
    """
    if media_file is None:
        yield "❌ Загрузите видео или аудио файл.", "", None
        return

    file_path = media_file if isinstance(media_file, str) else media_file.name

    if not is_media_file(file_path):
        yield f"❌ Это не видео/аудио: {Path(file_path).name}", "", None
        return

    name = _model_key(model_choice)

    # Модель не скачана — сначала качаем (с реальным прогрессом).
    if not is_model_cached(name):
        for status in _download_gen(name, progress):
            yield status, "", None
        if not is_model_cached(name):  # скачивание не удалось
            return

    yield f"🎧 Модель «{name}»: распознавание речи...", "", None

    def cb(frac, desc):
        progress(frac, desc=desc)

    text, debug_info = transcribe_media(file_path, name, progress_callback=cb)

    if text is None or not text.strip():
        yield f"❌ Не удалось распознать речь.\n\n🔍 {debug_info}", "", None
        return

    txt_path = transcript_path_for(file_path)
    srt_path = txt_path.with_suffix(".srt")
    saved = [str(p) for p in (txt_path, srt_path) if p.exists()]

    status = f"✅ Файл: {Path(file_path).name}\n\n{debug_info}"
    yield status, text, (saved or None)


# ──────────────────────────────────────────────
# Прошлые расшифровки (сохраняются на диск в output/)
# ──────────────────────────────────────────────

def _history_choices() -> list[tuple[str, str]]:
    """Список (подпись, путь) сохранённых расшифровок, свежие первыми."""
    import time
    out = []
    for p in list_transcripts():
        mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        out.append((f"{p.parent.name}  ·  {mtime}", str(p)))
    return out


def refresh_history():
    """Обновляет выпадающий список прошлых расшифровок."""
    choices = _history_choices()
    value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=value)


def open_transcript(path: str):
    """Открывает выбранную прошлую расшифровку (текст + файлы)."""
    if not path:
        return "", None
    text, files = read_transcript(path)
    return text, (files or None)


def load_initial():
    """При открытии страницы: последняя расшифровка + список прошлых."""
    choices = _history_choices()
    value = choices[0][1] if choices else None
    text, files = read_transcript(value) if value else ("", [])
    return gr.update(choices=choices, value=value), text, (files or None)


def create_app() -> gr.Blocks:
    """Создаёт и возвращает Gradio-приложение."""

    with gr.Blocks(title="Calls Recognize", theme=_build_theme(), css=CUSTOM_CSS) as app:

        gr.HTML("""
        <div class="header-text">
            <h1>📝 Calls Recognize</h1>
            <p>Расшифровка записей созвонов в текст</p>
            <p style="font-size: 0.85rem; margin-top: 0.5rem;">
                Whisper ASR • Русский и авто • Экспорт .txt и .srt (с таймкодами)
            </p>
        </div>
        """)

        gr.Markdown("### 🧠 Модель распознавания")
        gr.Markdown(
            "Выберите модель Whisper: крупнее — точнее, но медленнее и больше размер "
            "скачивания. Модель качается один раз и кешируется.\n\n"
            "**Скачивание идёт по одной** — повторный клик встанет в очередь. "
            "Прогресс загрузки (проценты/МБ) показывается **в полосе над кнопкой**, "
            "а текстовый статус — в поле «Статус модели» ниже."
        )
        with gr.Row():
            model_dd = gr.Dropdown(
                choices=list(WHISPER_MODELS.values()),
                value=WHISPER_MODELS.get(WHISPER_MODEL, list(WHISPER_MODELS.values())[2]),
                label="Модель Whisper",
                scale=3,
            )
            download_btn = gr.Button("⬇️ Скачать / загрузить", scale=1)
        model_status = gr.Textbox(
            label="Статус модели",
            lines=3,
            interactive=False,
            placeholder="Нажмите «Скачать / загрузить», чтобы подготовить модель заранее (необязательно)."
        )

        gr.Markdown("---")

        gr.Markdown("### 🎥 Запись созвона")
        gr.Markdown("""
        Загрузите **видео или аудио** записи созвона — речь будет распознана в текст выбранной моделью.

        **Форматы:** `.mp4` `.mov` `.mkv` `.webm` `.avi` `.mp3` `.wav` `.m4a` `.ogg` `.flac`

        ⏱️ Расшифровка идёт на CPU; длинные записи могут занять несколько минут.
        Первое использование новой модели — плюс время на её скачивание.

        💾 Результат сохраняется в `output/<имя_файла>/` как `.txt` и `.srt` (с таймкодами).
        """)
        media_input = gr.File(
            label="",
            file_types=[
                ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
                ".flv", ".wmv", ".mp3", ".wav", ".m4a", ".ogg",
                ".oga", ".opus", ".flac", ".aac",
            ],
            type="filepath",
        )
        transcribe_btn = gr.Button("📝 Расшифровать", variant="primary", size="lg")

        gr.Markdown("---")

        gr.Markdown("### 📄 Результат расшифровки")
        transcribe_status = gr.Textbox(
            label="Статус",
            lines=6,
            interactive=False,
            placeholder="Загрузите запись и нажмите «Расшифровать»..."
        )
        transcript_text = gr.Textbox(
            label="Распознанный текст (можно редактировать)",
            lines=16,
            interactive=True,
            placeholder="Здесь появится расшифровка речи..."
        )
        transcript_files = gr.File(
            label="📥 Скачать: текст (.txt) и субтитры с таймкодами (.srt)",
            file_count="multiple",
        )

        gr.Markdown("---")

        gr.Markdown("### 🕘 Прошлые расшифровки")
        gr.Markdown(
            "Все расшифровки сохраняются на диск и доступны после перезагрузки страницы. "
            "Выберите любую, чтобы снова открыть её текст и файлы."
        )
        with gr.Row():
            history_dd = gr.Dropdown(
                choices=[],
                label="Сохранённые расшифровки",
                scale=3,
            )
            refresh_btn = gr.Button("🔄 Обновить", scale=1)
        open_btn = gr.Button("📂 Открыть выбранную")

        # ── Обработчики истории ──
        refresh_btn.click(fn=refresh_history, outputs=[history_dd])
        open_btn.click(
            fn=open_transcript,
            inputs=[history_dd],
            outputs=[transcript_text, transcript_files],
        )

        # При открытии страницы — последняя расшифровка + список прошлых
        app.load(
            fn=load_initial,
            outputs=[history_dd, transcript_text, transcript_files],
        )

        # При выборе модели в дропдауне — сразу показываем, скачана она или нет
        model_dd.change(
            fn=model_status_wrapper,
            inputs=[model_dd],
            outputs=[model_status],
        )
        # При открытии страницы — статус модели по умолчанию
        app.load(
            fn=model_status_wrapper,
            inputs=[model_dd],
            outputs=[model_status],
        )

        # concurrency_limit=1 — скачивания сериализуются (по одной, очередью)
        download_btn.click(
            fn=download_model_wrapper,
            inputs=[model_dd],
            outputs=[model_status],
            concurrency_limit=1,
        )

        transcribe_btn.click(
            fn=transcribe_wrapper,
            inputs=[media_input, model_dd],
            outputs=[transcribe_status, transcript_text, transcript_files],
        ).then(
            fn=refresh_history,
            outputs=[history_dd],
        )

    app.queue()  # нужно для gr.Progress / track_tqdm
    return app
