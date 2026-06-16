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
    # produced by extractor.py / table extraction. Defaults to the real Ola
    # Electric DRHP extraction so a cold open shows genuine, page-cited
    # events (not a synthetic sample); the whole demo still runs with zero
    # API spend. Override with EVENTS_PATH in .env (e.g. the broader
    # fixtures/sample_events.json for full event-type coverage).
    events_path: str = "fixtures/ola_drhp_extracted.json"


settings = Settings()
