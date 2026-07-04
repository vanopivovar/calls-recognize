"""
Gradio-интерфейс Calls Recognize — расшифровка созвонов
"""

import gradio as gr
from pathlib import Path

from transcriber import (
    is_media_file,
    open_segments,
    save_transcription,
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


# CSS на переменных темы — кастомные элементы сами следуют светлой/тёмной теме.
CUSTOM_CSS = """
.gradio-container {
    max-width: 1060px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.app-title h2 { margin: 0; font-weight: 600; }
.app-title p { margin: 0; color: var(--body-text-color-subdued); font-size: 0.85rem; }
/* Карточки-группы: мягкая тень для разделения (особенно в светлой теме) */
.gr-group, .panel-card {
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}
"""

# JS: переключение темы мгновенно, без перезагрузки (не прерывает процесс).
THEME_TOGGLE_JS = "() => { document.body.classList.toggle('dark'); }"
# JS при загрузке: стартуем в тёмной теме по умолчанию.
INIT_DARK_JS = "() => { document.body.classList.add('dark'); }"


def _build_theme() -> gr.themes.Base:
    """Тема со светлой и тёмной палитрой (переключается классом .dark)."""
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        # Светлая палитра: мягкий серый фон страницы, белые блоки, заметные границы
        body_background_fill="#eef1f5",
        block_background_fill="#ffffff",
        block_border_color="#d5dbe2",
        border_color_primary="#d5dbe2",
        input_background_fill="#ffffff",
        input_border_color="#cfd6de",
        # Тёмная палитра
        body_background_fill_dark="#1a1d24",
        block_background_fill_dark="#252a33",
        block_border_color_dark="#353b47",
        border_color_primary_dark="#353b47",
        input_background_fill_dark="#2d3440",
        input_border_color_dark="#353b47",
        # Основная кнопка — одинаковая в обеих темах
        button_primary_background_fill="#4a6785",
        button_primary_background_fill_hover="#5b7c99",
        button_primary_text_color="#ffffff",
        button_primary_background_fill_dark="#4a6785",
        button_primary_background_fill_hover_dark="#5b7c99",
    )


def _default_model() -> str:
    """Ключ модели по умолчанию."""
    return WHISPER_MODEL if WHISPER_MODEL in WHISPER_MODELS else "small"


def _model_key(choice: str) -> str:
    """Значение дропдауна — уже ключ модели; подстраховка для дефолта."""
    return choice or _default_model()


def _model_choices() -> list[tuple[str, str]]:
    """
    Пары (подпись, ключ) для дропдауна. Подпись содержит метку:
    ✅ — модель уже скачана, ⬇️ — ещё нет.
    """
    out = []
    for key, label in WHISPER_MODELS.items():
        mark = "✅" if is_model_cached(key) else "⬇️"
        out.append((f"{mark} {label}", key))
    return out


def refresh_models(selected: str):
    """Пересобирает список моделей с актуальными метками ✅/⬇️, сохраняя выбор."""
    return gr.update(choices=_model_choices(), value=selected or _default_model())


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


def transcribe_wrapper(media_file, model_choice: str, line_per_segment: bool, progress=gr.Progress()):
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

    yield f"🎧 Модель «{name}»: распознавание речи…", "", None

    segments, info, meta = open_segments(file_path, name)
    if segments is None:
        yield f"❌ Не удалось начать распознавание.\n\n🔍 {meta}", "", None
        return

    duration = getattr(info, "duration", None)
    # Цикл по сегментам — здесь Gradio может прервать процесс кнопкой «Стоп».
    timed: list[tuple[float, float, str]] = []
    for seg in segments:
        seg_text = (seg.text or "").strip()
        if seg_text:
            timed.append((seg.start, seg.end, seg_text))
        if duration:
            try:
                frac = min((seg.end or 0.0) / duration, 0.99)
                progress(frac, desc=f"Распознано {seg.end:.0f}/{duration:.0f} сек")
            except Exception:
                pass

    if not timed:
        yield "❌ Речь не распознана (тишина или нет звуковой дорожки).", "", None
        return

    text = save_transcription(file_path, timed, line_per_segment)
    txt_path = transcript_path_for(file_path)
    saved = [str(p) for p in (txt_path, txt_path.with_suffix(".srt")) if p.exists()]

    lang = getattr(info, "language", None)
    status = (
        f"✅ Файл: {Path(file_path).name}\n"
        f"Фрагментов: {len(timed)}" + (f" · язык: {lang}" if lang else "")
    )
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

    with gr.Blocks(
        title="Calls Recognize",
        theme=_build_theme(),
        css=CUSTOM_CSS,
        js=INIT_DARK_JS,   # по умолчанию — тёмная тема
    ) as app:

        # ── Верхняя панель: название слева, переключатель темы справа ──
        with gr.Row():
            with gr.Column(scale=8):
                gr.HTML(
                    '<div class="app-title">'
                    '<h2>📝 Calls Recognize</h2>'
                    '<p>Расшифровка записей созвонов · Whisper ASR · экспорт .txt и .srt</p>'
                    '</div>'
                )
            with gr.Column(scale=1, min_width=90):
                theme_btn = gr.Button("🌗", size="sm")
        theme_btn.click(fn=None, inputs=None, outputs=None, js=THEME_TOGGLE_JS)

        with gr.Tabs():

            # ═════════════ Вкладка: Расшифровка ═════════════
            with gr.TabItem("🎙️ Расшифровка"):
                with gr.Row(equal_height=True):

                    # ЛЕВО — вход
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Запись созвона")
                            media_input = gr.File(
                                label="Видео или аудио",
                                file_types=[
                                    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v",
                                    ".flv", ".wmv", ".mp3", ".wav", ".m4a", ".ogg",
                                    ".oga", ".opus", ".flac", ".aac",
                                ],
                                type="filepath",
                            )
                            model_dd = gr.Dropdown(
                                choices=_model_choices(),
                                value=_default_model(),
                                label="Модель для распознавания",
                                info="Крупнее — точнее, но медленнее. Скачать модели — во вкладке «Модели».",
                            )
                            line_per_segment_cb = gr.Checkbox(
                                label="Каждая реплика с новой строки",
                                value=False,
                            )
                            with gr.Row():
                                transcribe_btn = gr.Button(
                                    "Расшифровать", variant="primary", scale=3, interactive=False
                                )
                                stop_transcribe_btn = gr.Button(
                                    "⏹ Стоп", variant="stop", scale=1, visible=False
                                )
                            gr.Markdown(
                                "<sub>Кнопка активна после загрузки файла. Нескачанная модель "
                                "скачается автоматически.</sub>"
                            )

                    # ПРАВО — результат
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Результат")
                            transcribe_status = gr.Textbox(
                                label="Статус",
                                lines=2,
                                interactive=False,
                                placeholder="Загрузите запись и нажмите «Расшифровать»…",
                            )
                            transcript_text = gr.Textbox(
                                label="Распознанный текст",
                                lines=15,
                                interactive=True,
                                show_copy_button=True,
                                placeholder="Здесь появится расшифровка…",
                            )
                            transcript_files = gr.File(
                                label="Файлы: .txt и .srt",
                                file_count="multiple",
                            )

                # История: открытие в один клик
                with gr.Accordion("Прошлые расшифровки", open=True):
                    with gr.Row():
                        history_dd = gr.Dropdown(
                            choices=[],
                            label="Выберите запись, чтобы открыть",
                            scale=5,
                        )
                        refresh_btn = gr.Button("Обновить", scale=1, min_width=120)

            # ═════════════ Вкладка: Модели ═════════════
            with gr.TabItem("🧠 Модели"):
                gr.Markdown("### Управление моделями")
                gr.Markdown(
                    "Скачайте нужные модели заранее. Модель качается один раз и кешируется. "
                    "Крупнее — точнее, но медленнее и больше размер. Скачивание идёт по одной."
                )
                mgmt_dd = gr.Dropdown(
                    choices=_model_choices(),
                    value=_default_model(),
                    label="Модель для скачивания",
                )
                with gr.Row():
                    download_btn = gr.Button("Скачать", variant="primary", scale=3)
                    stop_download_btn = gr.Button(
                        "⏹ Стоп", variant="stop", scale=1, visible=False
                    )
                model_status = gr.Textbox(
                    label="Статус модели",
                    lines=2,
                    interactive=False,
                    placeholder="Выберите модель и нажмите «Скачать».",
                )

        # ══════════════ Обработчики ══════════════

        _show_stop = lambda: (gr.update(visible=False), gr.update(visible=True))
        _show_action = lambda: (gr.update(visible=True), gr.update(visible=False))

        # Кнопка «Расшифровать» активна только когда выбран файл
        media_input.change(
            fn=lambda f: gr.update(interactive=bool(f)),
            inputs=[media_input],
            outputs=[transcribe_btn],
        )

        # Статус модели при выборе во вкладке «Модели»
        mgmt_dd.change(fn=model_status_wrapper, inputs=[mgmt_dd], outputs=[model_status])

        # ── Скачивание: показать «Стоп», качать, обновить метки, вернуть кнопку ──
        dl_event = download_btn.click(
            fn=_show_stop, outputs=[download_btn, stop_download_btn]
        ).then(
            fn=download_model_wrapper,
            inputs=[mgmt_dd],
            outputs=[model_status],
            concurrency_limit=1,
        )
        dl_event.then(fn=refresh_models, inputs=[mgmt_dd], outputs=[mgmt_dd]) \
                .then(fn=refresh_models, inputs=[model_dd], outputs=[model_dd]) \
                .then(fn=_show_action, outputs=[download_btn, stop_download_btn])
        stop_download_btn.click(
            fn=lambda: ("⏹ Остановлено. Частично скачанное догрузится позже.",
                        gr.update(visible=True), gr.update(visible=False)),
            outputs=[model_status, download_btn, stop_download_btn],
            cancels=[dl_event],
        )

        # ── Расшифровка: показать «Стоп», распознать, обновить, вернуть кнопку ──
        tr_event = transcribe_btn.click(
            fn=_show_stop, outputs=[transcribe_btn, stop_transcribe_btn]
        ).then(
            fn=transcribe_wrapper,
            inputs=[media_input, model_dd, line_per_segment_cb],
            outputs=[transcribe_status, transcript_text, transcript_files],
            show_progress_on=[transcribe_status],
        )
        tr_event.then(fn=refresh_history, outputs=[history_dd]) \
                .then(fn=refresh_models, inputs=[model_dd], outputs=[model_dd]) \
                .then(fn=refresh_models, inputs=[mgmt_dd], outputs=[mgmt_dd]) \
                .then(fn=_show_action, outputs=[transcribe_btn, stop_transcribe_btn])
        stop_transcribe_btn.click(
            fn=lambda: ("⏹ Расшифровка остановлена.",
                        gr.update(visible=True), gr.update(visible=False)),
            outputs=[transcribe_status, transcribe_btn, stop_transcribe_btn],
            cancels=[tr_event],
        )

        # История: открытие выбранной записи в один клик
        history_dd.change(
            fn=open_transcript,
            inputs=[history_dd],
            outputs=[transcript_text, transcript_files],
        )
        refresh_btn.click(fn=refresh_history, outputs=[history_dd])

        # При открытии страницы: история + последняя расшифровка, статус и метки моделей
        app.load(
            fn=load_initial,
            outputs=[history_dd, transcript_text, transcript_files],
        )
        app.load(fn=model_status_wrapper, inputs=[mgmt_dd], outputs=[model_status])
        app.load(fn=refresh_models, inputs=[model_dd], outputs=[model_dd])
        app.load(fn=refresh_models, inputs=[mgmt_dd], outputs=[mgmt_dd])

    app.queue()  # нужно для gr.Progress / track_tqdm
    return app
