"""
Calls Recognize — расшифровка записей созвонов в текст (Whisper ASR)

Точка входа приложения.
"""

from ui import create_app

# Тема и CSS задаются внутри create_app() на уровне gr.Blocks.
app = create_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
    )
