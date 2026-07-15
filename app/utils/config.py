"""Load environment settings and define project filesystem locations."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
VECTOR_DB_DIR = DATA_DIR / "vectordb"
CHROMA_COLLECTION_NAME = "morocco_tourism"

load_dotenv(dotenv_path=ENV_FILE)
GOOGLE_API_KEY = (os.getenv("GOOGLE_API_KEY") or "").strip()
GEMINI_CHAT_MODEL = (
    (os.getenv("GEMINI_CHAT_MODEL") or "").strip()
    or "gemini-3.5-flash"
)


class ConfigurationError(RuntimeError):
    """Indicate that a required application setting is missing or invalid."""


def ensure_data_directories() -> None:
    """Create the document and vector database directories when absent."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


def get_google_api_key() -> str:
    """Return the configured Gemini API key or raise a clear error."""
    api_key = GOOGLE_API_KEY
    if not api_key:
        raise ConfigurationError(
            "GOOGLE_API_KEY is not configured. Copy .env.example to .env "
            "and add your Google Gemini API key."
        )
    return api_key


ensure_data_directories()
