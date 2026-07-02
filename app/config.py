"""
Configuration: loads all environment variables and constants.
Uses pydantic-settings for type-safe env parsing.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenRouter / LLM
    openrouter_api_key: str = ""
    use_llm: bool = True
    llm_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-small"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Server
    port: int = 8000
    host: str = "0.0.0.0"

    # Retrieval
    top_k: int = 10
    chroma_db_path: str = str(BASE_DIR / "app" / "data" / "chroma_db")
    catalog_path: str = str(BASE_DIR / "app" / "data" / "catalog.json")
    bm25_index_path: str = str(BASE_DIR / "app" / "data" / "bm25_index.pkl")

    # Scraping
    scrape_delay_seconds: float = 1.0

    # Retrieval tuning
    rrf_k: int = 60          # RRF constant
    dense_weight: float = 0.6
    bm25_weight: float = 0.4

    # Agent behavior
    min_facts_for_recommendation: int = 2   # role + one of seniority/context


@lru_cache
def get_settings() -> Settings:
    return Settings()
