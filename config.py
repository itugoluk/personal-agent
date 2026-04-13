import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MINIMAX_API_BASE = "https://api.minimax.io/v1"

HISTORY_DIR = os.path.expanduser("~/.olympus")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")
MAX_OUTPUT_CHARS = 8000

# Available models keyed by shorthand
MODELS = {
    "groq-llama": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "label": "Llama 3.3 70B (Groq)",
    },
    "groq-fast": {
        "provider": "groq",
        "model_id": "llama-3.1-8b-instant",
        "label": "Llama 3.1 8B (Groq)",
    },
    "sonnet": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "label": "Claude Sonnet 4.6",
    },
    "opus": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-6",
        "label": "Claude Opus 4.6",
    },
    "minimax": {
        "provider": "minimax",
        "model_id": "MiniMax-Text-01",
        "label": "MiniMax M2.7",
    },
}

DEFAULT_MODEL = "groq-llama"
