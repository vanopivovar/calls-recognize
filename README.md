# Calls Recognize

Расшифровка записей созвонов в текст на русском языке. Загружаете видео или
аудио — получаете распознанный текст и субтитры с таймкодами (`.srt`).

Построено на [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(CTranslate2) — работает на CPU, не требует GPU/torch.

## Возможности

- Видео и аудио: `.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac` и др.
- Видео читается напрямую (встроенный ffmpeg), отдельно извлекать аудио не нужно.
- Экспорт в `.txt` (сплошной текст) и `.srt` (с таймкодами).
- Результат сохраняется в `output/<имя_файла>/`.
- Русский по умолчанию, есть автоопределение языка.
- VAD-фильтр тишины — точнее на записях созвонов.

## Запуск через Docker

```bash
docker compose up -d --build
```

Откройте <http://localhost:7861>.

> При первом запуске скачивается модель Whisper (~0.5 ГБ для `small`),
> дальше работает офлайн. Модель кешируется в томе `whisper_cache`.

## Запуск локально (Python)

```bash
pip install -r requirements.txt
python app.py
```

Нужен установленный `ffmpeg` в системе. Откройте <http://localhost:7861>.

## Настройки (переменные окружения)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `small` / `medium` / `large-v3` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` (быстро) / `int8_float16` / `float32` |
| `WHISPER_LANGUAGE` | `ru` | `ru` или `auto` |
| `WHISPER_DOWNLOAD_ROOT` | — | Куда качать модель |

Пример — точнее, но медленнее:

```bash
WHISPER_MODEL=medium WHISPER_LANGUAGE=auto python app.py
```
