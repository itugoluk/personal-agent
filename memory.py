import json
import os
from config import HISTORY_DIR, HISTORY_FILE


def load() -> list:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save(messages: list) -> None:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(messages, f, indent=2)


def clear() -> None:
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
