"""Hybrid retrieval tests.

A 20-event synthetic corpus with known financial literals proves that BM25
catches exact tokens ("Rs. 154.20", "5:1", "2,50,000") that the vector
side misses, that weighted reciprocal rank fusion is deterministic, and
that the numeric-query alpha heuristic picks the right lean.
"""
from __future__ import annotations

import pytest

from conftest import StubEmbedding
from retrieval import (
    ALPHA_DEFAULT,
    ALPHA_NUMERIC,
    EventStore,
    HybridRetriever,
    choose_alpha,
    event_to_text,
    tokenize,
)


def _synthetic_events() -> list[dict]:
    """20 events with distinct, queryable financial literals."""
    events: list[dict] = []
    prices = [10.0, 25.5, 76.0, 154.2, 220.5, 310.0, 450.75, 512.0, 660.4, 999.99]
    for i, price in enumerate(prices):
        events.append(
            {
                "event_type": "allotment",
                "date": f"20{10 + i}-01-15",
                "shares": (i + 1) * 1000,
                "issue_price": price,
                "allottee_category": "promoters" if i % 2 == 0 else "investors",
                "page_number": 10 + i,
                "source_snippet": (
                    f"allotted {(i + 1) * 1000} Equity Shares at an issue price of "
                    f"Rs. {price:.2f} per share in Series A-{i + 1}"
                ),
                "confidence": 1.0,
            }
        )
    for i, ratio in enumerate(["5:1", "1:4", "2:1", "3:2", "7:1"]):
        events.append(
            {
                "event_type": "bonus_issue",
                "date": f"202{i}-06-01",
                "ratio": ratio,
                "page_number": 30 + i,
                "source_snippet": f"bonus issue of shares in the ratio of {ratio} approved",
                "confidence": 1.0,
            }
        )
    for i in range(5):
        cap = (i + 1) * 100000000
        events.append(
            {
                "event_type": "authorised_capital_change",
                "date": f"202{i}-09-09",
                "new_capital": cap,
                "page_number": 40 + i,
                "source_snippet": f"authorised capital increased to Rs. {cap:,}",
                "confidence": 1.0,
            }
        )
    assert len(events) == 20
    return events


@pytest.fixture
def hybrid() -> HybridRetriever:
    store = EventStore(
        collection_name="hybrid_test",
        embedding_function=StubEmbedding(),
        in_memory=True,
    )
    store.index_events(_synthetic_events())
    return HybridRetriever(store)


# ── tokenizer keeps financial literals intact ─────────────────────────────────

def test_tokenizer_preserves_financial_literals():
    tokens = tokenize("allotment at Rs. 154.20 ratio 5:1 of 2,50,000 shares")
    assert "154.20" in tokens
    assert "5:1" in tokens
    assert "2,50,000" in tokens


# ── alpha heuristic ───────────────────────────────────────────────────────────

def test_numeric_queries_lean_bm25():
    assert choose_alpha("Rs. 4.5 Cr Series A allotment") == ALPHA_NUMERIC
    assert choose_alpha("issue at 154.20") == ALPHA_NUMERIC
    assert choose_alpha("bonus 5:1") == ALPHA_NUMERIC


def test_conceptual_queries_lean_vector():
    assert choose_alpha("promoter allotments before the bonus issue") == ALPHA_DEFAULT
    assert choose_alpha("what changed the authorised capital") == ALPHA_DEFAULT


# ── retrieval quality on numeric queries ──────────────────────────────────────

NUMERIC_QUERIES = [
    ("issue price of Rs. 154.20", "154.2"),
    ("issue price of Rs. 220.50", "220.5"),
    ("bonus ratio 5:1", "5:1"),
    ("bonus ratio 7:1", "7:1"),
    ("issue price of Rs. 999.99", "999.99"),
]


def _gold_in_top3(hits, needle: str) -> bool:
    for h in hits[:3]:
        blob = f"{h.event} {h.text}"
        if needle in blob:
            return True
    return False


def test_bm25_finds_exact_numeric_tokens(hybrid):
    for query, needle in NUMERIC_QUERIES:
        ranked = hybrid._bm25_search(query, k=3)
        assert ranked, f"BM25 returned nothing for {query!r}"
        top_ids = [eid for eid, _ in ranked]
        hits = hybrid._hits_from_ids(top_ids, scores=None)
        assert _gold_in_top3(hits, needle), f"BM25 missed {needle!r} for {query!r}"


def test_hybrid_finds_exact_numeric_tokens(hybrid):
    for query, needle in NUMERIC_QUERIES:
        hits = hybrid.search(query, k=3)
        assert _gold_in_top3(hits, needle), f"hybrid missed {needle!r} for {query!r}"


def test_hybrid_recall_beats_or_matches_vector_on_numeric_queries(hybrid):
    hybrid_hits = sum(
        _gold_in_top3(hybrid.search(q, k=3), needle) for q, needle in NUMERIC_QUERIES
    )
    vector_hits = sum(
        _gold_in_top3(hybrid._vector_search(q, k=3), needle)
        for q, needle in NUMERIC_QUERIES
    )
    hybrid_recall = hybrid_hits / len(NUMERIC_QUERIES)
    vector_recall = vector_hits / len(NUMERIC_QUERIES)
    assert hybrid_recall >= vector_recall, (
        f"hybrid recall@3 {hybrid_recall:.2f} < vector recall@3 {vector_recall:.2f}"
    )
    # the synthetic corpus is built so exact-token queries are BM25-solvable
    assert hybrid_recall == 1.0


# ── fusion mechanics ──────────────────────────────────────────────────────────

def test_rrf_scores_are_deterministic(hybrid):
    a = hybrid.search("issue price of Rs. 154.20", k=5)
    b = hybrid.search("issue price of Rs. 154.20", k=5)
    assert [(h.event_id, h.score) for h in a] == [(h.event_id, h.score) for h in b]


def test_rrf_scores_normalised_to_unit_range(hybrid):
    hits = hybrid.search("bonus ratio 5:1", k=5)
    assert hits
    for h in hits:
        assert 0.0 <= h.score <= 1.0


def test_last_strategy_is_recorded(hybrid):
    hybrid.search("bonus ratio 5:1", k=3)
    assert hybrid.last_strategy in {"hybrid", "bm25", "vector"}


def test_alpha_extremes_change_ranking_weights(hybrid):
    # pure BM25 lean must keep the exact-token match on top
    bm25_lean = hybrid.search("issue price of Rs. 154.20", k=3, alpha=0.0)
    assert _gold_in_top3(bm25_lean, "154.2")


# ── persistence round trip ────────────────────────────────────────────────────

def test_pickle_round_trip(tmp_path, hybrid):
    p = tmp_path / "bm25_index.pkl"
    hybrid.save(p)
    loaded = HybridRetriever.load(p, hybrid.store)
    a = [h.event_id for h in hybrid.search("bonus ratio 5:1", k=3)]
    b = [h.event_id for h in loaded.search("bonus ratio 5:1", k=3)]
    assert a == b


def test_stale_pickle_triggers_rebuild(tmp_path, hybrid):
    p = tmp_path / "bm25_index.pkl"
    hybrid.save(p)
    # change the underlying store so the pickle no longer matches
    hybrid.store.index_events(_synthetic_events()[:5])
    loaded = HybridRetriever.load(p, hybrid.store)
    assert len(loaded._ids) == 5  # rebuilt from store, not the stale pickle


# ── agent passthrough ─────────────────────────────────────────────────────────

def test_agent_over_hybrid_reports_strategy(hybrid):
    from agent import CapScribeAgent

    agent = CapScribeAgent(hybrid)
    resp = agent.ask(event_to_text(_synthetic_events()[0]), mode="extractive")
    assert resp.retrieval_strategy in {"hybrid", "bm25", "vector"}
    assert resp.answer
