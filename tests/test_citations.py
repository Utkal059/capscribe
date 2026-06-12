"""Page-level citation tests.

Covers the provenance chain end to end: page-tagged chunking in the
extractor, snippet validation against source pages, citation metadata
surviving the vector-store round trip, and /ask responses carrying page
citations that never exceed the document's page count and are never
invented.
"""
from __future__ import annotations

import json
from pathlib import Path

from agent import CapScribeAgent
from extractor import (
    FALLBACK_CONFIDENCE,
    attach_provenance,
    make_chunks,
    snippet_match_ratio,
)
from schema import SearchHit, citation_from_hit

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "sample_events.json"


# ── page-tagged chunking ──────────────────────────────────────────────────────

def test_make_chunks_tags_every_page():
    pages = ["alpha text", "beta text", "gamma text"]
    chunks = make_chunks(pages, first_page=1)
    joined = "\n".join(text for _, _, text in chunks)
    for n in (1, 2, 3):
        assert f"[PAGE {n}]" in joined


def test_make_chunks_honours_start_page_offset():
    pages = ["first", "second"]
    chunks = make_chunks(pages, first_page=40)
    start_pg, end_pg, text = chunks[0]
    assert start_pg == 40
    assert "[PAGE 40]" in text
    assert "[PAGE 1]" not in text


# ── snippet fuzzy validation ──────────────────────────────────────────────────

PAGE_TEXT = (
    "Pursuant to a resolution dated February 3, 2017, our Company allotted "
    "10,000 Equity Shares of face value Rs. 10 each at an issue price of "
    "Rs. 10 per share for cash to the Promoters."
)


def test_snippet_exact_substring_scores_one():
    assert snippet_match_ratio("our Company allotted 10,000 Equity Shares", PAGE_TEXT) == 1.0


def test_snippet_whitespace_differences_still_match():
    snippet = "our  Company\nallotted 10,000   Equity Shares"
    assert snippet_match_ratio(snippet, PAGE_TEXT) == 1.0


def test_snippet_fabricated_text_scores_low():
    assert snippet_match_ratio("the quick brown fox jumps over the lazy dog", PAGE_TEXT) < 0.5


def test_attach_provenance_keeps_genuine_snippet():
    events = [
        {
            "event_type": "allotment",
            "page_number": 3,
            "source_snippet": "our Company allotted 10,000 Equity Shares",
        }
    ]
    out = attach_provenance(events, {3: PAGE_TEXT})
    assert out[0]["source_snippet"] == "our Company allotted 10,000 Equity Shares"
    assert out[0]["confidence"] == 1.0


def test_attach_provenance_replaces_fabricated_snippet():
    events = [
        {
            "event_type": "allotment",
            "page_number": 3,
            "source_snippet": "completely invented text that is not on the page at all",
        }
    ]
    out = attach_provenance(events, {3: PAGE_TEXT})
    # Snippet replaced with genuine page text, confidence downgraded.
    assert out[0]["source_snippet"].startswith("Pursuant to a resolution")
    assert out[0]["confidence"] == FALLBACK_CONFIDENCE


def test_attach_provenance_never_invents_pages():
    events = [
        {"event_type": "allotment", "page_number": 99, "source_snippet": "anything"},
        {"event_type": "allotment", "page_number": "junk", "source_snippet": "anything"},
    ]
    out = attach_provenance(events, {3: PAGE_TEXT})
    for ev in out:
        assert ev["page_number"] is None
        assert "source_snippet" not in ev


# ── fixture coverage ──────────────────────────────────────────────────────────

def test_fixture_events_have_full_citation_coverage(sample_events):
    total_pages = json.loads(FIXTURE.read_text(encoding="utf-8"))["total_pages"]
    for ev in sample_events:
        assert ev.get("page_number") is not None, f"missing page_number: {ev['event_type']}"
        assert ev.get("source_snippet"), f"missing source_snippet: {ev['event_type']}"
        assert 1 <= ev["page_number"] <= total_pages


# ── retrieval round trip ──────────────────────────────────────────────────────

def test_search_hits_carry_provenance(store):
    hits = store.search("bonus issue", k=store.count())
    assert hits
    for h in hits:
        assert h.event_id is not None
        assert h.event.get("page_number") is not None
        assert h.event.get("source_snippet")


# ── ask response citations ────────────────────────────────────────────────────

def test_ask_returns_page_citations(store):
    total_pages = json.loads(FIXTURE.read_text(encoding="utf-8"))["total_pages"]
    agent = CapScribeAgent(store)
    resp = agent.ask("bonus issue ratio", mode="extractive")
    assert resp.page_citations, "expected page citations for grounded answer"
    assert len(resp.page_citations) == len(resp.citations)
    for cit in resp.page_citations:
        assert 1 <= cit.page_number <= total_pages
        assert cit.source_snippet
        assert cit.event_id
    assert resp.query_time_ms > 0
    assert resp.retrieval_strategy in {"vector", "bm25", "hybrid"}


def test_citation_from_hit_requires_provenance():
    bare = SearchHit(event={"event_type": "allotment"}, score=0.9, text="x")
    assert citation_from_hit(bare) is None
    good = SearchHit(
        event={"event_type": "allotment", "page_number": 7, "source_snippet": "verbatim"},
        score=0.9,
        text="x",
        event_id="event-0",
    )
    cit = citation_from_hit(good)
    assert cit is not None and cit.page_number == 7 and cit.event_id == "event-0"
