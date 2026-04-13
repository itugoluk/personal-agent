import os
from dotenv import load_dotenv

load_dotenv()

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_API_BASE = "https://api.minimax.io/v1"
MODEL = "MiniMax-Text-01"
HISTORY_DIR = os.path.expanduser("~/.personal-agent")
HISTORY_FILE = os.path.join(HISTORY_DIR, "history.json")
MAX_OUTPUT_CHARS = 8000
