"""
Конфигурация Calls Recognize
"""

import os
from pathlib import Path

# Куда сохраняются расшифровки (output/<имя>/<имя>.txt|.srt|meta.json)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Папка-источник: сюда кладут записи, чтобы расшифровывать файл/папку целиком
INPUT_DIR = Path(os.environ.get("INPUT_DIR", "input"))
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# Настройки пользователя (какие модели доступны и т.п.)
SETTINGS_PATH = Path(os.environ.get("SETTINGS_PATH", str(OUTPUT_DIR / "settings.json")))
