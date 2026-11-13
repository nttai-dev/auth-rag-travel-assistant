"""Load Firebase and Ollama configuration from the local .env file."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Read .env beside this module so running from any directory resolves the same config.
load_dotenv(Path(__file__).with_name(".env"), override=False)

FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "").strip()
FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY", "").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "").strip()
EMBED_MODEL = os.getenv("EMBED_MODEL", "").strip() or "embeddinggemma"
CHAT_MODEL = os.getenv("CHAT_MODEL", "").strip() or "qwen3:4b"
