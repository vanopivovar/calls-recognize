"""
Calls Recognize — расшифровка записей созвонов в текст (Whisper ASR)

Точка входа. Поднимает Gradio UI и встроенный MCP-сервер
(эндпоинт /gradio_api/mcp/sse), чтобы ИИ-агенты могли вызывать сервис.
"""

import os

from ui import create_app

app = create_app()

if __name__ == "__main__":
    app.launch(
        # По умолчанию сервис доступен только с этого компьютера: пароля у него нет.
        # В Docker адрес задаётся переменной (0.0.0.0), а наружу порт открывает compose.
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7861")),
        share=False,
        show_error=True,
        mcp_server=True,
    )
