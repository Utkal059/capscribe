"""Vector + BM25 hybrid search over extracted capital events.

Closes the "vector store + retrieval" gap: events are rendered to natural
language, embedded, and stored in a persistent Chroma collection so an
analyst can semantically interrogate a filing ("show me promoter
allotments before the bonus issue") instead of grepping JSON.

Vector-only search misses exact financial tokens ("Rs. 154.20", "5:1",
"2,50,000") that embeddings blur together, so `HybridRetriever` runs BM25
over the same corpus (including the verbatim source snippets) and fuses
both rankings with reciprocal rank fusion. A query-shape heuristic leans
the fusion toward BM25 for numeric queries and toward vectors for
conceptual ones.

Embeddings default to Chroma's built-in local ONNX model
(all-MiniLM-L6-v2): no API key, no torch, no network after first download.
The embedding function is injectable so tests run fully offline.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api.types import EmbeddingFunction
from rank_bm25 import BM25Okapi

from config import settings
from schema import SearchHit

logger = logging.getLogger("capscribe.retrieval")


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
    # extended event types
    if event.get("amount_per_share"):
        parts.append(f"dividend {event['amount_per_share']} per share")
    if event.get("shares_bought_back"):
        parts.append(f"bought back {event['shares_bought_back']} shares")
    if event.get("warrants_exercised"):
        parts.append(f"{event['warrants_exercised']} warrants exercised")
    if event.get("exercise_price"):
        parts.append(f"exercise price {event['exercise_price']}")
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
            # Chroma metadata only accepts scalars; non-scalar fields (bbox
            # lists, nested dicts) are dropped from the index but remain in
            # the source JSON.
            metas.append(
                {
                    k: v
                    for k, v in ev.items()
                    if v is not None and isinstance(v, (str, int, float, bool))
                }
            )
        self._collection.add(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def search(self, query: str, k: Optional[int] = None) -> list[SearchHit]:
        k = k or settings.top_k
        n = min(k, max(self.count(), 1))
        res = self._collection.query(query_texts=[query], n_results=n)
        hits: list[SearchHit] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for hit_id, doc, meta, dist in zip(ids, docs, metas, dists):
            # Chroma returns L2/cosine distance; convert to a 0..1 similarity.
            score = round(1.0 / (1.0 + float(dist)), 4)
            hits.append(SearchHit(event=dict(meta), score=score, text=doc, event_id=hit_id))
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


# ── Hybrid retrieval (BM25 + vector + reciprocal rank fusion) ──────────────────

# Tokens keep internal ./,/: so financial literals survive intact:
# "154.20", "5:1", "2,50,000", "1,00,00,000".
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,:][a-z0-9]+)*")

# Queries that look numeric/identifier-heavy lean BM25 (low alpha):
# amounts with units (4.5 Cr / 10 L / 3M / 12%), bare numbers, ratios,
# ISIN-style identifiers.
_NUMERIC_QUERY_RE = re.compile(
    r"(\d+\.?\d*\s*(cr|crore|l|lakh|m|mn|b|bn|%))|(\d[\d,.]*)|(\b\d+:\d+\b)|(\bin[a-z0-9]{10}\b)",
    re.IGNORECASE,
)

ALPHA_NUMERIC = 0.3   # lean BM25 for numeric / identifier queries
ALPHA_DEFAULT = 0.6   # lean vector for conceptual queries
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def choose_alpha(query: str) -> float:
    """Heuristic fusion weight: 0 = pure BM25, 1 = pure vector."""
    return ALPHA_NUMERIC if _NUMERIC_QUERY_RE.search(query) else ALPHA_DEFAULT


class HybridRetriever:
    """BM25 + vector search fused with weighted reciprocal rank fusion.

    Wraps an :class:`EventStore` (vector side) and a BM25 index built over
    the same documents *plus* their verbatim source snippets, so exact
    financial tokens are matchable even when embeddings blur them.

    `last_strategy` records how the most recent query was answered
    ("hybrid", "vector" or "bm25") for observability.
    """

    def __init__(self, store: EventStore) -> None:
        self.store = store
        self.last_strategy = "hybrid"
        self._ids: list[str] = []
        self._docs: list[str] = []
        self._metas: list[dict] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._build_from_store()

    # -- index construction ----------------------------------------------------

    def _build_from_store(self) -> None:
        res = self.store._collection.get()
        self._ids = list(res.get("ids", []) or [])
        self._docs = list(res.get("documents", []) or [])
        self._metas = [dict(m) for m in (res.get("metadatas", []) or [])]
        self._corpus_tokens = [
            tokenize(f"{doc} {meta.get('source_snippet', '')}")
            for doc, meta in zip(self._docs, self._metas)
        ]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        logger.info("hybrid: BM25 index built over %d documents", len(self._ids))

    # -- persistence -------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the BM25 corpus so startup can skip re-tokenising."""
        payload = {
            "ids": self._ids,
            "docs": self._docs,
            "metas": self._metas,
            "corpus_tokens": self._corpus_tokens,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: str | Path, store: EventStore) -> "HybridRetriever":
        """Load a serialised index; falls back to rebuilding when stale."""
        self = cls.__new__(cls)
        self.store = store
        self.last_strategy = "hybrid"
        payload = pickle.loads(Path(path).read_bytes())
        self._ids = payload["ids"]
        self._docs = payload["docs"]
        self._metas = payload["metas"]
        self._corpus_tokens = payload["corpus_tokens"]
        if len(self._ids) != store.count():
            logger.warning("hybrid: pickle stale (%d vs %d docs); rebuilding",
                           len(self._ids), store.count())
            self._build_from_store()
            return self
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        return self

    # -- search ------------------------------------------------------------------

    def _bm25_search(self, query: str, k: int) -> list[tuple[str, float]]:
        """Return [(event_id, bm25_score)] sorted best-first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._ids, scores), key=lambda p: (-p[1], p[0]))
        return [(eid, float(s)) for eid, s in ranked[:k] if s > 0]

    def _vector_search(self, query: str, k: int) -> list[SearchHit]:
        return self.store.search(query, k=k)

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        alpha: Optional[float] = None,
    ) -> list[SearchHit]:
        """Weighted reciprocal rank fusion of vector and BM25 rankings.

        score(d) = alpha * 1/(RRF_K + rank_vec(d)) + (1-alpha) * 1/(RRF_K + rank_bm25(d))

        Reported scores are normalised by the maximum achievable fused
        score (rank 1 in both lists), so a document that tops both
        rankings scores 1.0.
        """
        k = k or settings.top_k
        if alpha is None:
            alpha = choose_alpha(query)
            logger.info("hybrid: auto alpha=%.1f for %r", alpha, query)

        vec_hits = self._vector_search(query, k * 2)
        bm25_hits = self._bm25_search(query, k * 2)

        if not bm25_hits and not vec_hits:
            self.last_strategy = "hybrid"
            return []
        if not bm25_hits:
            self.last_strategy = "vector"
            return vec_hits[:k]
        if not vec_hits:
            self.last_strategy = "bm25"
            return self._hits_from_ids([eid for eid, _ in bm25_hits[:k]],
                                       scores=None)
        self.last_strategy = "hybrid"

        fused: dict[str, float] = {}
        for rank, hit in enumerate(vec_hits, start=1):
            if hit.event_id:
                fused[hit.event_id] = fused.get(hit.event_id, 0.0) + alpha / (RRF_K + rank)
        for rank, (eid, _) in enumerate(bm25_hits, start=1):
            fused[eid] = fused.get(eid, 0.0) + (1 - alpha) / (RRF_K + rank)

        max_possible = alpha / (RRF_K + 1) + (1 - alpha) / (RRF_K + 1)
        ranked_ids = sorted(fused, key=lambda eid: (-fused[eid], eid))[:k]
        return self._hits_from_ids(
            ranked_ids,
            scores={eid: round(fused[eid] / max_possible, 4) for eid in ranked_ids},
        )

    def _hits_from_ids(
        self, ids: list[str], scores: Optional[dict[str, float]]
    ) -> list[SearchHit]:
        by_id = {eid: i for i, eid in enumerate(self._ids)}
        hits: list[SearchHit] = []
        for rank, eid in enumerate(ids, start=1):
            i = by_id.get(eid)
            if i is None:
                continue
            score = scores[eid] if scores else round(1.0 / (1.0 + rank * 0.1), 4)
            hits.append(
                SearchHit(event=dict(self._metas[i]), score=score,
                          text=self._docs[i], event_id=eid)
            )
        return hits

    # -- passthroughs so the agent/API can treat this as the store ---------------

    def count(self) -> int:
        return self.store.count()

    def stats(self) -> dict:
        return self.store.stats()
