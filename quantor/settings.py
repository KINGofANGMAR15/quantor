"""Сохранение настроек «Квантора» (режим углов, геометрия окна).

Файл: ~/.quantor/settings.json. Никаких реестров и внешних хранилищ.
"""

import json
import os
from typing import Any, Dict

_DIR = os.path.join(os.path.expanduser("~"), ".quantor")
_PATH = os.path.join(_DIR, "settings.json")


def load_settings() -> Dict[str, Any]:
    try:
        with open(_PATH, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(settings: Dict[str, Any]) -> None:
    try:
        os.makedirs(_DIR, exist_ok=True)
        with open(_PATH, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass  # настройки — не критично; калькулятор работает и без них
