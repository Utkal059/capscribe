"""Shared test fixtures.

Tests use a deterministic, offline embedding function and an in-memory
Chroma store, so the suite never downloads a model, never hits the network,
and never calls the Anthropic API. It runs in well under a second.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from retrieval import EventStore

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_events.json"


class StubEmbedding(EmbeddingFunction):
    """Hash-based embedding: deterministic and dependency-free.

    Identical text maps to an identical vector, so exact-match queries are
    reliably retrievable in tests without a real semantic model.
    """

    def __init__(self) -> None:  # noqa: D401 - required by newer chroma
        pass

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        vectors: Embeddings = []
        for text in input:
            digest = hashlib.sha256(text.lower().encode()).digest()
            vectors.append([b / 255.0 for b in digest[:32]])
        return vectors

    def name(self) -> str:  # required by chroma >= 1.x
        return "stub-sha256"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "StubEmbedding":
        return StubEmbedding()


@pytest.fixture
def sample_events() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["capital_events"]


@pytest.fixture
def store(sample_events) -> EventStore:
    s = EventStore(
        collection_name="capital_events_test",
        embedding_function=StubEmbedding(),
        in_memory=True,
    )
    s.index_events(sample_events)
    return s
