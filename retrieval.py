"""Vector search over extracted capital events.

Closes the "vector store + retrieval" gap: events are rendered to natural
language, embedded, and stored in a persistent Chroma collection so an
analyst can semantically interrogate a filing ("show me promoter
allotments before the bonus issue") instead of grepping JSON.

Embeddings default to Chroma's built-in local ONNX model
(all-MiniLM-L6-v2): no API key, no torch, no network after first download.
The embedding function is injectable so tests run fully offline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api.types import EmbeddingFunction

from config import settings
from schema import SearchHit


def event_to_text(event: dict) -> str:
    """Render an event as a search-friendly sentence."""
    et = event.get("event_type", "event")
    parts = [et.replace("_", " ")]
    if event.get("date"):
        parts.append(f"on {event['date']}")
    if event.get("allottee_category"):
        parts.append(f"to {event['allottee_category']}")
    if event.get("shares"):
        parts.append(f"{event['shares']} shares")
    if event.get("ratio"):
        parts.append(f"ratio {event['ratio']}")
    if event.get("issue_price"):
        parts.append(f"at price {event['issue_price']}")
    if event.get("consideration"):
        parts.append(f"for {event['consideration']}")
    if event.get("new_capital"):
        parts.append(
            f"authorised capital {event.get('old_capital')} to {event['new_capital']}"
        )
    if event.get("resolution_type"):
        parts.append(f"via {event['resolution_type']}")
    return ", ".join(str(p) for p in parts)


class EventStore:
    """Thin wrapper around a Chroma collection of capital events."""

    def __init__(
        self,
        path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[EmbeddingFunction] = None,
        in_memory: bool = False,
    ) -> None:
        self.collection_name = collection_name or settings.collection_name
        if in_memory:
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=path or settings.chroma_path)
        kwargs = {"name": self.collection_name}
        if embedding_function is not None:
            kwargs["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(**kwargs)

    def index_events(self, events: list[dict]) -> int:
        """Replace the collection contents with the given events. Returns count."""
        # Fresh build: drop and recreate so re-indexing is idempotent.
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        ef = getattr(self._collection, "_embedding_function", None)
        kwargs = {"name": self.collection_name}
        if ef is not None:
            kwargs["embedding_function"] = ef
        self._collection = self._client.get_or_create_collection(**kwargs)

        if not events:
            return 0
        ids, docs, metas = [], [], []
        for i, ev in enumerate(events):
            ids.append(f"event-{i}")
            docs.append(event_to_text(ev))
            metas.append({k: v for k, v in ev.items() if v is not None})
        self._collection.add(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def search(self, query: str, k: Optional[int] = None) -> list[SearchHit]:
        k = k or settings.top_k
        n = min(k, max(self.count(), 1))
        res = self._collection.query(query_texts=[query], n_results=n)
        hits: list[SearchHit] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            # Chroma returns L2/cosine distance; convert to a 0..1 similarity.
            score = round(1.0 / (1.0 + float(dist)), 4)
            hits.append(SearchHit(event=dict(meta), score=score, text=doc))
        return hits

    def count(self) -> int:
        return self._collection.count()

    def stats(self) -> dict:
        """Counts by event type, computed from stored metadata."""
        res = self._collection.get()
        counts: dict[str, int] = {}
        for meta in res.get("metadatas", []) or []:
            et = meta.get("event_type", "unknown")
            counts[et] = counts.get(et, 0) + 1
        return {"total": self.count(), "by_type": counts}


def load_events(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data.get("capital_events", [])
