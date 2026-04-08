import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openweather_api_key: str = os.getenv("OPENWEATHER_API_KEY", "")
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    faiss_index_dir: str = os.getenv(
        "FAISS_INDEX_DIR",
        str(Path(__file__).resolve().parents[2] / "data" / "faiss_store"),
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    crawl4ai_enabled: bool = os.getenv(
        "CRAWL4AI_ENABLED",
        "false" if sys.platform.startswith("win") else "true",
    ).lower() in {"1", "true", "yes"}


settings = Settings()
