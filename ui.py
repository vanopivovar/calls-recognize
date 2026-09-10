"""
Gradio-интерфейс Calls Recognize — расшифровка созвонов.

Вся логика — в service.py / transcriber.py; здесь только компоновка,
обработчики и регистрация API-эндпоинтов (они же — MCP-инструменты).
"""

import threading
import time
from pathlib import Path

import gradio as gr

import service as S
from config import INPUT_DIR
from transcriber import (
    WHISPER_MODEL,
    WHISPER_MODELS,
    downloaded_bytes,
    download_model_files,
    enabled_models,
    ensure_model,
    is_model_cached,
    model_status_text,
    model_total_bytes,
)


# ──────────────────────────────────────────────
# Тема / стили
# ──────────────────────────────────────────────

CUSTOM_CSS = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.app-title h2 { margin: 0; font-weight: 600; }
.app-title p { margin: 0; color: var(--body-text-color-subdued); font-size: 0.85rem; }
.gr-group, .panel-card { box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05); }
"""

THEME_TOGGLE_JS = "() => { document.body.classList.toggle('dark'); }"
INIT_DARK_JS = "() => { document.body.classList.add('dark'); }"

MEDIA_TYPES = [
    ".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".flv", ".wmv",
    ".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac",
]


def _build_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#eef1f5",
        block_background_fill="#ffffff",
        block_border_color="#d5dbe2",
        border_color_primary="#d5dbe2",
        input_background_fill="#ffffff",
        input_border_color="#cfd6de",
        body_background_fill_dark="#1a1d24",
        block_background_fill_dark="#252a33",
        block_border_color_dark="#353b47",
        border_color_primary_dark="#353b47",
        input_background_fill_dark="#2d3440",
        input_border_color_dark="#353b47",
        button_primary_background_fill="#4a6785",
        button_primary_background_fill_hover="#5b7c99",
        button_primary_text_color="#ffffff",
        button_primary_background_fill_dark="#4a6785",
        button_primary_background_fill_hover_dark="#5b7c99",
    )


# ──────────────────────────────────────────────
# Модели: выбор, метки, скачивание, таблица
# ──────────────────────────────────────────────

def _default_model() -> str:
    en = enabled_models()
    return WHISPER_MODEL if WHISPER_MODEL in en else (en[0] if en else "small")


def _model_choices(only_enabled: bool = True) -> list[tuple[str, str]]:
    names = enabled_models() if only_enabled else list(WHISPER_MODELS)
    return [
        (f"{'✅' if is_model_cached(k) else '⬇️'} {WHISPER_MODELS[k]}", k) for k in names
    ]


def refresh_models(selected: str, only_enabled: bool = True):
    ch = _model_choices(only_enabled)
    vals = [v for _, v in ch]
    val = selected if selected in vals else (vals[0] if vals else None)
    return gr.update(choices=ch, value=val)


def refresh_models_enabled(selected: str):
    return refresh_models(selected, only_enabled=True)


def refresh_models_all(selected: str):
    return refresh_models(selected, only_enabled=False)


def _download_gen(name: str, progress):
    """Скачивание в фоне + опрос реального размера на диске раз в секунду."""
    if is_model_cached(name):
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
            yield f"⏳ Скачиваю «{name}»: {done/mb:.0f} МБ…"
        time.sleep(1.0)
    th.join()
    if err:
        yield f"❌ Не удалось скачать «{name}»: {err['e']}"
        return
    ensure_model(name)
    yield model_status_text(name)


def download_model_wrapper(model_choice: str, progress=gr.Progress()):
    yield from _download_gen(model_choice or _default_model(), progress)


def delete_model_wrapper(model_choice: str) -> str:
    r = S.delete_model(model_choice or _default_model())
    return ("✅ " if r["ok"] else "❌ ") + r["message"]


def mgmt_buttons(model_choice: str):
    cached = is_model_cached(model_choice or _default_model())
    return gr.update(visible=not cached), gr.update(visible=False), gr.update(visible=cached)


def mgmt_select(model_choice: str):
    return (model_status_text(model_choice or _default_model()), *mgmt_buttons(model_choice))


def stop_download_ui(model_choice: str):
    d, s, x = mgmt_buttons(model_choice)
    return "⏹ Остановлено. Частично скачанное догрузится позже.", d, s, x


TABLE_HEADERS = ["Модель", "Скачана", "МБ", "Обновление", "Доступна"]


def models_table(check_updates: bool = False) -> list[list]:
    rows = S.list_models(check_updates=check_updates)

    def upd(u):
        return "—" if u is None else ("🔄 есть" if u else "нет")

    return [
        [r["name"], "✅" if r["downloaded"] else "⬇️", r["size_mb"] or "",
         upd(r["update_available"]), "✔" if r["enabled"] else ""]
        for r in rows
    ]


def check_updates_ui():
    return models_table(check_updates=True), "🔄 Проверено. «есть» — на Hugging Face новее версия: удалите и скачайте заново."


def save_enabled_ui(names: list[str]):
    r = S.set_enabled_models(names)
    return ("✅ " if r["ok"] else "❌ ") + r["message"], models_table(False)


# ──────────────────────────────────────────────
# Расшифровка (файлы, папки, батч)
# ──────────────────────────────────────────────

def _collect_sources(uploaded, explorer) -> list[str]:
    paths: list[str] = []
    for u in (uploaded or []):
        paths.append(u if isinstance(u, str) else u.name)
    for e in (explorer or []):
        p = Path(e)
        if not p.is_absolute():
            p = INPUT_DIR / p
        paths.append(str(p))
    return paths


def transcribe_wrapper(uploaded, explorer, model_choice: str, line_per_segment: bool,
                       progress=gr.Progress()):
    """Батч: каждый файл → отдельный результат. Выдаёт (лог, текст последнего, файлы)."""
    sources = _collect_sources(uploaded, explorer)
    if not sources:
        yield "❌ Выберите файл(ы) или папку.", "", None
        return

    files: list[str] = []
    for s in sources:
        files += [str(f) for f in S.iter_media_files(s)]
    files = list(dict.fromkeys(files))
    if not files:
        yield "❌ В выбранном нет видео/аудио.", "", None
        return

    name = model_choice or _default_model()
    if not is_model_cached(name):
        for st in _download_gen(name, progress):
            yield st, "", None
        if not is_model_cached(name):
            return

    n = len(files)
    log: list[str] = []
    produced: list[str] = []
    last_text = ""
    for i, f in enumerate(files, 1):
        log.append(f"⏳ [{i}/{n}] {Path(f).name}")
        yield "\n".join(log), last_text, (produced or None)

        def cb(frac, desc, _i=i):
            progress((_i - 1 + frac) / n, desc=f"[{_i}/{n}] {desc}")

        r = S.transcribe_file(f, name, line_per_segment, progress_callback=cb)
        if r["ok"]:
            log[-1] = (f"✅ [{i}/{n}] {Path(f).name} — {r['segments']} фрагм., "
                       f"{r.get('duration') or 0:.0f} сек")
            produced += [r["txt"], r["srt"]]
            last_text = r["text"]
        else:
            log[-1] = f"❌ [{i}/{n}] {Path(f).name} — {r['error']}"
        yield "\n".join(log), last_text, (produced or None)

    ok = sum(1 for l in log if l.startswith("✅"))
    log.append(f"Готово: {ok}/{n}. Файлы — в output/<имя>/")
    yield "\n".join(log), last_text, (produced or None)


# ──────────────────────────────────────────────
# История
# ──────────────────────────────────────────────

def _history_choices() -> list[tuple[str, str]]:
    out = []
    for r in S.list_transcripts():
        extra = f"  ·  {r['created']}" + (f"  ·  {r['model']}" if r.get("model") else "")
        out.append((f"{r['name']}{extra}", r["txt"]))
    return out


def refresh_history():
    ch = _history_choices()
    return gr.update(choices=ch, value=(ch[0][1] if ch else None))


def open_transcript(path: str):
    if not path:
        return "", None
    r = S.read_transcript(path)
    return r["text"], (r["files"] or None)


def load_initial():
    ch = _history_choices()
    v = ch[0][1] if ch else None
    if v:
        r = S.read_transcript(v)
        return gr.update(choices=ch, value=v), r["text"], (r["files"] or None)
    return gr.update(choices=ch, value=None), "", None


# ──────────────────────────────────────────────
# Публичный API (имена функций = имена MCP-инструментов)
# ──────────────────────────────────────────────

def list_models(check_updates: bool = False) -> list[dict]:
    """Список моделей Whisper: downloaded, size_mb, enabled, update_available (если check_updates=true — сверка с Hugging Face)."""
    return S.api_list_models(check_updates)


def download_model(name: str) -> dict:
    """Скачать модель Whisper по имени (tiny, base, small, medium, large-v2, large-v3, large-v3-turbo)."""
    return S.api_download_model(name)


def delete_model(name: str) -> dict:
    """Удалить скачанную модель из кеша по имени."""
    return S.api_delete_model(name)


def set_enabled_models(names: list[str]) -> dict:
    """Задать список моделей, доступных для выбора и скачивания."""
    return S.api_set_enabled_models(names)


def transcribe(path: str, model: str = "", line_per_segment: bool = False) -> list[dict]:
    """Расшифровать файл или папку (путь внутри контейнера /app/input/... или относительно input/). Один файл = один результат: txt, srt, text, model, language, duration."""
    return S.api_transcribe(path, model, line_per_segment)


def list_transcripts() -> list[dict]:
    """Список сохранённых расшифровок с метаданными (name, txt, srt, created, model, duration)."""
    return S.api_list_transcripts()


def read_transcript(name: str) -> dict:
    """Текст и файлы расшифровки по имени (папка в output/) или пути к .txt."""
    return S.api_read_transcript(name)


# ──────────────────────────────────────────────
# Приложение
# ──────────────────────────────────────────────

def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="Calls Recognize",
        theme=_build_theme(),
        css=CUSTOM_CSS,
        js=INIT_DARK_JS,
    ) as app:

        # ── Верхняя панель ──
        with gr.Row():
            with gr.Column(scale=8):
                gr.HTML(
                    '<div class="app-title">'
                    '<h2>📝 Calls Recognize</h2>'
                    '<p>Расшифровка записей созвонов · Whisper ASR · экспорт .txt и .srt · MCP</p>'
                    '</div>'
                )
            with gr.Column(scale=1, min_width=90):
                theme_btn = gr.Button("🌗", size="sm")
        theme_btn.click(fn=None, inputs=None, outputs=None, js=THEME_TOGGLE_JS,
                        api_name=False)

        with gr.Tabs():

            # ═════════════ Вкладка: Расшифровка ═════════════
            with gr.TabItem("🎙️ Расшифровка"):
                with gr.Row(equal_height=False):

                    # ЛЕВО — источник
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Источник")
                            with gr.Tabs():
                                with gr.TabItem("Загрузить файлы"):
                                    media_input = gr.File(
                                        label="Видео или аудио (можно несколько)",
                                        file_types=MEDIA_TYPES,
                                        file_count="multiple",
                                        type="filepath",
                                    )
                                with gr.TabItem("Из папки input"):
                                    gr.Markdown(
                                        f"<sub>Положите записи в `{INPUT_DIR}` (на хосте — папка "
                                        "`input/` проекта). Выберите файлы или целую папку.</sub>"
                                    )
                                    explorer = gr.FileExplorer(
                                        root_dir=str(INPUT_DIR),
                                        glob="**/*",
                                        file_count="multiple",
                                        label="Файлы и папки в input/",
                                        height=240,
                                    )
                                    explorer_refresh = gr.Button("Обновить список", size="sm")
                            model_dd = gr.Dropdown(
                                choices=_model_choices(True),
                                value=_default_model(),
                                label="Модель для распознавания",
                                info="Только включённые модели (настраивается во вкладке «Модели»).",
                            )
                            line_per_segment_cb = gr.Checkbox(
                                label="Каждая реплика с новой строки", value=False,
                            )
                            with gr.Row():
                                transcribe_btn = gr.Button(
                                    "Расшифровать", variant="primary", scale=3, interactive=False
                                )
                                stop_transcribe_btn = gr.Button(
                                    "⏹ Стоп", variant="stop", scale=1, visible=False
                                )
                            gr.Markdown(
                                "<sub>Один файл = один результат в `output/<имя>/`. "
                                "Нескачанная модель скачается автоматически.</sub>"
                            )

                    # ПРАВО — результат
                    with gr.Column(scale=1):
                        with gr.Group():
                            gr.Markdown("### Результат")
                            transcribe_status = gr.Textbox(
                                label="Ход обработки",
                                lines=6,
                                interactive=False,
                                placeholder="Выберите источник и нажмите «Расшифровать»…",
                            )
                            transcript_text = gr.Textbox(
                                label="Текст (последний обработанный файл)",
                                lines=14,
                                interactive=True,
                                show_copy_button=True,
                                placeholder="Здесь появится расшифровка…",
                            )
                            transcript_files = gr.File(
                                label="Файлы: .txt и .srt (все обработанные)",
                                file_count="multiple",
                            )

                with gr.Accordion("Прошлые расшифровки", open=True):
                    with gr.Row():
                        history_dd = gr.Dropdown(
                            choices=[], label="Выберите запись, чтобы открыть", scale=5,
                        )
                        refresh_btn = gr.Button("Обновить", scale=1, min_width=120)

            # ═════════════ Вкладка: Модели ═════════════
            with gr.TabItem("🧠 Модели"):
                gr.Markdown("### Состояние моделей")
                models_df = gr.Dataframe(
                    headers=TABLE_HEADERS,
                    value=models_table(False),
                    interactive=False,
                    wrap=True,
                )
                with gr.Row():
                    check_updates_btn = gr.Button("🔄 Проверить обновления")
                    table_refresh_btn = gr.Button("Обновить таблицу")
                updates_status = gr.Textbox(label="", show_label=False, interactive=False,
                                            placeholder="Проверка обновлений сравнивает версию на диске с Hugging Face.")

                gr.Markdown("### Доступные модели")
                enabled_cg = gr.CheckboxGroup(
                    choices=list(WHISPER_MODELS),
                    value=enabled_models(),
                    label="Какие модели показывать для выбора и скачивания",
                )
                save_enabled_btn = gr.Button("Сохранить набор")
                enabled_status = gr.Textbox(label="", show_label=False, interactive=False)

                gr.Markdown("### Скачать / удалить")
                mgmt_dd = gr.Dropdown(
                    choices=_model_choices(False),
                    value=_default_model(),
                    label="Модель",
                )
                with gr.Row():
                    download_btn = gr.Button("⬇️ Скачать", variant="primary", scale=3)
                    stop_download_btn = gr.Button("⏹ Стоп", variant="stop", scale=1, visible=False)
                    delete_btn = gr.Button("🗑 Удалить", variant="stop", scale=1, visible=False)
                model_status = gr.Textbox(
                    label="Статус модели", lines=2, interactive=False,
                    placeholder="Скачанную можно удалить, нескачанную — скачать.",
                )

            # ═════════════ Вкладка: API / MCP ═════════════
            with gr.TabItem("🔌 API / MCP"):
                gr.Markdown(
                    "### Подключение ИИ-агента (MCP)\n"
                    "Сервис публикует MCP-сервер по адресу **`/gradio_api/mcp/sse`** "
                    "(например `http://localhost:7861/gradio_api/mcp/sse`).\n\n"
                    "Инструменты: `list_models`, `download_model`, `delete_model`, "
                    "`set_enabled_models`, `transcribe`, `list_transcripts`, `read_transcript`.\n\n"
                    "**Claude Code:**\n"
                    "```bash\nclaude mcp add --transport sse calls-recognize "
                    "http://localhost:7861/gradio_api/mcp/sse\n```\n"
                    "**Claude Desktop / другие клиенты** (через `mcp-remote`):\n"
                    "```json\n{ \"mcpServers\": { \"calls-recognize\": {\n"
                    "  \"command\": \"npx\",\n"
                    "  \"args\": [\"mcp-remote\", \"http://localhost:7861/gradio_api/mcp/sse\"]\n"
                    "} } }\n```\n"
                    "Схема HTTP-API: `/gradio_api/openapi.json`. Пути для `transcribe` — внутри "
                    "контейнера (`/app/input/...`) или относительно папки `input/`."
                )

        # ══════════════ Обработчики (api_name=False — не светим в API/MCP) ══════════════

        _show_stop = lambda: (gr.update(visible=False), gr.update(visible=True))
        _show_action = lambda: (gr.update(visible=True), gr.update(visible=False))

        def _has_sources(uploaded, explorer):
            return gr.update(interactive=bool(uploaded) or bool(explorer))

        media_input.change(fn=_has_sources, inputs=[media_input, explorer],
                           outputs=[transcribe_btn], api_name=False)
        explorer.change(fn=_has_sources, inputs=[media_input, explorer],
                        outputs=[transcribe_btn], api_name=False)
        explorer_refresh.click(fn=lambda: gr.update(root_dir=str(INPUT_DIR)),
                               outputs=[explorer], api_name=False)

        # Модели: выбор → статус + кнопки
        mgmt_dd.change(fn=mgmt_select, inputs=[mgmt_dd],
                       outputs=[model_status, download_btn, stop_download_btn, delete_btn],
                       api_name=False)

        dl_event = download_btn.click(
            fn=_show_stop, outputs=[download_btn, stop_download_btn], api_name=False,
        ).then(
            fn=download_model_wrapper, inputs=[mgmt_dd], outputs=[model_status],
            concurrency_limit=1, api_name=False,
        )
        dl_event.then(fn=refresh_models_all, inputs=[mgmt_dd], outputs=[mgmt_dd], api_name=False) \
                .then(fn=refresh_models_enabled, inputs=[model_dd], outputs=[model_dd], api_name=False) \
                .then(fn=lambda: models_table(False), outputs=[models_df], api_name=False) \
                .then(fn=mgmt_buttons, inputs=[mgmt_dd],
                      outputs=[download_btn, stop_download_btn, delete_btn], api_name=False)
        stop_download_btn.click(
            fn=stop_download_ui, inputs=[mgmt_dd],
            outputs=[model_status, download_btn, stop_download_btn, delete_btn],
            cancels=[dl_event], api_name=False,
        )

        delete_btn.click(fn=delete_model_wrapper, inputs=[mgmt_dd], outputs=[model_status],
                         api_name=False) \
            .then(fn=refresh_models_all, inputs=[mgmt_dd], outputs=[mgmt_dd], api_name=False) \
            .then(fn=refresh_models_enabled, inputs=[model_dd], outputs=[model_dd], api_name=False) \
            .then(fn=lambda: models_table(False), outputs=[models_df], api_name=False) \
            .then(fn=mgmt_buttons, inputs=[mgmt_dd],
                  outputs=[download_btn, stop_download_btn, delete_btn], api_name=False)

        check_updates_btn.click(fn=check_updates_ui, outputs=[models_df, updates_status],
                                api_name=False)
        table_refresh_btn.click(fn=lambda: models_table(False), outputs=[models_df],
                                api_name=False)
        save_enabled_btn.click(fn=save_enabled_ui, inputs=[enabled_cg],
                               outputs=[enabled_status, models_df], api_name=False) \
            .then(fn=refresh_models_enabled, inputs=[model_dd], outputs=[model_dd], api_name=False)

        # Расшифровка (батч)
        tr_event = transcribe_btn.click(
            fn=_show_stop, outputs=[transcribe_btn, stop_transcribe_btn], api_name=False,
        ).then(
            fn=transcribe_wrapper,
            inputs=[media_input, explorer, model_dd, line_per_segment_cb],
            outputs=[transcribe_status, transcript_text, transcript_files],
            show_progress_on=[transcribe_status], api_name=False,
        )
        tr_event.then(fn=refresh_history, outputs=[history_dd], api_name=False) \
                .then(fn=refresh_models_enabled, inputs=[model_dd], outputs=[model_dd], api_name=False) \
                .then(fn=refresh_models_all, inputs=[mgmt_dd], outputs=[mgmt_dd], api_name=False) \
                .then(fn=lambda: models_table(False), outputs=[models_df], api_name=False) \
                .then(fn=_show_action, outputs=[transcribe_btn, stop_transcribe_btn], api_name=False)
        stop_transcribe_btn.click(
            fn=lambda: ("⏹ Расшифровка остановлена.", gr.update(visible=True),
                        gr.update(visible=False)),
            outputs=[transcribe_status, transcribe_btn, stop_transcribe_btn],
            cancels=[tr_event], api_name=False,
        )

        # История
        history_dd.change(fn=open_transcript, inputs=[history_dd],
                          outputs=[transcript_text, transcript_files], api_name=False)
        refresh_btn.click(fn=refresh_history, outputs=[history_dd], api_name=False)

        # Загрузка страницы
        app.load(fn=load_initial, outputs=[history_dd, transcript_text, transcript_files],
                 api_name=False)
        app.load(fn=mgmt_select, inputs=[mgmt_dd],
                 outputs=[model_status, download_btn, stop_download_btn, delete_btn],
                 api_name=False)
        app.load(fn=refresh_models_enabled, inputs=[model_dd], outputs=[model_dd], api_name=False)
        app.load(fn=refresh_models_all, inputs=[mgmt_dd], outputs=[mgmt_dd], api_name=False)

        # ══════════════ Публичный API → MCP-инструменты ══════════════
        gr.api(list_models, api_name="list_models")
        gr.api(download_model, api_name="download_model")
        gr.api(delete_model, api_name="delete_model")
        gr.api(set_enabled_models, api_name="set_enabled_models")
        gr.api(transcribe, api_name="transcribe")
        gr.api(list_transcripts, api_name="list_transcripts")
        gr.api(read_transcript, api_name="read_transcript")

    app.queue()
    return app
