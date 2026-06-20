"""Centralised, typed configuration.

Every tunable lives here so the system stays observable: one place to see
what model is used, where the index is persisted, and which document is
currently loaded. Values are read from the environment / .env file.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Anthropic (only used for live extraction and llm-mode answers) ---
    anthropic_api_key: str = ""
    # Cheapest current model — keeps credit burn minimal.
    extract_model: str = "claude-haiku-4-5-20251001"
    answer_model: str = "claude-haiku-4-5-20251001"
    chunk_size: int = 2
    chunk_overlap: int = 1

    # --- Retrieval / vector store ---
    chroma_path: str = ".chroma"
    collection_name: str = "capital_events"
    top_k: int = 5

    # --- Data ---
    # The extracted JSON the API loads on startup. Point this at any file
    # produced by extractor.py. Defaults to the bundled fixture so the whole
    # demo runs with zero API spend.
    events_path: str = "fixtures/sample_events.json"


settings = Settings()
