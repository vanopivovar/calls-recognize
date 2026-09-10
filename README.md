# Calls Recognize

Расшифровка записей созвонов в текст на русском языке. Загружаете видео или
аудио (или указываете папку) — получаете распознанный текст и субтитры с
таймкодами (`.srt`). Есть встроенный **MCP-сервер**, чтобы ИИ-агент мог
управлять сервисом.

Построено на [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2) — работает на CPU, не требует GPU/torch.

## Возможности

- **Файл или папка.** Один файл = один результат: `output/<имя>/<имя>.txt`,
  `.srt` и `meta.json` (модель, язык, длительность, дата). Папка обходится
  рекурсивно; можно загрузить несколько файлов через браузер или положить их
  в `input/` и выбрать в проводнике.
- **Модели Whisper.** Таблица: скачана / размер / **есть обновление** на
  Hugging Face / доступна. Скачать, удалить, проверить обновления, выбрать
  какие модели показывать.
- **MCP / HTTP-API.** Инструменты `list_models`, `download_model`,
  `delete_model`, `set_enabled_models`, `transcribe`, `list_transcripts`,
  `read_transcript`.
- Видео читается напрямую (встроенный ffmpeg). Русский по умолчанию, есть
  автоопределение языка, VAD-фильтр тишины, реальный прогресс, кнопки «Стоп».
- Экспорт: сплошной текст или «каждая реплика с новой строки».

## Запуск через Docker

```bash
docker compose up -d --build
```

Откройте <http://localhost:7861>.

- `input/` — сюда кладите записи для расшифровки папкой/файлом.
- `output/` — сюда сохраняются результаты.
- Модели кешируются в томе `whisper_cache` (качаются один раз).

## Подключение ИИ-агента (MCP)

MCP-сервер доступен по адресу `http://localhost:7861/gradio_api/mcp/sse`.

**Claude Code:**

```bash
claude mcp add --transport sse calls-recognize http://localhost:7861/gradio_api/mcp/sse
```

**Claude Desktop / другие клиенты** (через `mcp-remote`):

```json
{
  "mcpServers": {
    "calls-recognize": {
      "command": "npx",
      "args": ["mcp-remote", "http://localhost:7861/gradio_api/mcp/sse"]
    }
  }
}
```

Пути для `transcribe` — внутри контейнера (`/app/input/meeting.mp4`) или
относительно папки `input/` (`meeting.mp4`, `2026-09/`). Схема HTTP-API:
`http://localhost:7861/gradio_api/openapi.json`.

## Запуск локально (Python)

```bash
pip install -r requirements.txt
python app.py
```

Нужен установленный `ffmpeg`. Откройте <http://localhost:7861>.

## Настройки (переменные окружения)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `WHISPER_MODEL` | `small` | модель по умолчанию |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` (быстро) / `int8_float16` / `float32` |
| `WHISPER_LANGUAGE` | `ru` | `ru` или `auto` |
| `WHISPER_DOWNLOAD_ROOT` | — | куда качать модели |
| `INPUT_DIR` | `input` | папка с записями |
| `OUTPUT_DIR` | `output` | папка результатов |
| `HF_TOKEN` | — | токен Hugging Face (быстрее скачивание, без троттлинга) |

Набор доступных моделей хранится в `output/settings.json`.
