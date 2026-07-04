"""
Gradio-интерфейс Calls Recognize — расшифровка созвонов
"""

import gradio as gr
from pathlib import Path

from transcriber import is_media_file, transcribe_media, transcript_path_for


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


def transcribe_wrapper(media_file, progress=gr.Progress(track_tqdm=False)):
    """
    Расшифровка видео/аудио в текст.
    Возвращает: (статус, распознанный_текст, [файлы .txt/.srt]).
    """
    if media_file is None:
        return "❌ Загрузите видео или аудио файл.", "", None

    file_path = media_file if isinstance(media_file, str) else media_file.name

    if not is_media_file(file_path):
        return f"❌ Это не видео/аудио: {Path(file_path).name}", "", None

    progress(0.1, desc="Распознавание речи (может занять несколько минут)...")
    text, debug_info = transcribe_media(file_path)
    progress(1.0, desc="Готово")

    if text is None or not text.strip():
        return f"❌ Не удалось распознать речь.\n\n🔍 {debug_info}", "", None

    txt_path = transcript_path_for(file_path)
    srt_path = txt_path.with_suffix(".srt")
    saved = [str(p) for p in (txt_path, srt_path) if p.exists()]

    status = f"✅ Файл: {Path(file_path).name}\n\n{debug_info}"
    return status, text, (saved or None)


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

        gr.Markdown("### 🎥 Запись созвона")
        gr.Markdown("""
        Загрузите **видео или аудио** записи созвона — речь будет распознана в текст (Whisper).

        **Форматы:** `.mp4` `.mov` `.mkv` `.webm` `.avi` `.mp3` `.wav` `.m4a` `.ogg` `.flac`

        ⏱️ Первый запуск качает модель Whisper (~0.5 ГБ). Расшифровка идёт на CPU,
        длинные записи могут занять несколько минут.

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

        transcribe_btn.click(
            fn=transcribe_wrapper,
            inputs=[media_input],
            outputs=[transcribe_status, transcript_text, transcript_files],
        )

    return app
