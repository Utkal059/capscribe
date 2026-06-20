"""Shared data models for CapScribe.

A single permissive `CapitalEvent` model covers all four event families
(allotment, bonus_issue, rights_issue, authorised_capital_change) because
each family carries a different field set. `extra="allow"` keeps any
unmodelled fields rather than dropping them, while the typed fields below
give validation and editor support for the common ones.

Provenance fields (page_number, source_snippet, confidence) make every
event citable back to the exact page and verbatim text it came from.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "allotment",
    "bonus_issue",
    "rights_issue",
    "authorised_capital_change",
    # extended coverage — annual reports, financial statements, diligence files
    "dividend_declaration",
    "share_repurchase",
    "warrant_exercise",
]

ExtractionMethod = Literal["table", "text", "llm", "ocr"]


class SourceProvenance(BaseModel):
    """Structured citation layer for an event.

    Backwards-compatible mirror of the flat provenance fields below. It is
    optional (defaults to ``None``) so existing extracted JSON and existing
    Chroma collections remain valid; when present it records *how* an event
    was extracted, which the evaluation harness breaks down per method.
    """

    page: int | None = None                       # 1-indexed source page
    section: str | None = None                     # nearest preceding heading
    bbox: list[float] | None = None               # [x0, y0, x1, y1] for tables
    extraction_method: ExtractionMethod = "llm"   # table | text | llm | ocr
    confidence: float = 1.0                        # 0.0-1.0


class CapitalEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_type: EventType
    date: str | None = None

    # provenance — every event traces back to a page and a verbatim excerpt
    page_number: int | None = None     # 1-indexed PDF page the event was found on
    source_snippet: str | None = None  # verbatim 1-2 sentence excerpt from that page
    bbox: list[float] | None = None    # [x0, y0, x1, y1] when table-sourced
    confidence: float = 1.0            # 0.0-1.0; lowered when provenance is weak
    # structured citation; optional + backwards-compatible (default None)
    source_provenance: SourceProvenance | None = None

    # allotment
    shares: int | None = None
    face_value: float | None = None
    issue_price: float | None = None
    consideration: str | None = None
    allottee_category: str | None = None

    # bonus / rights
    ratio: str | None = None
    shares_issued: int | None = None
    pre_issue_capital: int | None = None
    post_issue_capital: int | None = None
    price: float | None = None
    shares_offered: int | None = None

    # authorised capital change
    old_capital: int | None = None
    new_capital: int | None = None
    resolution_type: str | None = None

    # dividend_declaration
    amount_per_share: float | None = None
    record_date: str | None = None
    payment_date: str | None = None
    total_outflow: int | None = None

    # share_repurchase
    shares_bought_back: int | None = None
    remaining_buyback_authority: int | None = None

    # warrant_exercise
    warrants_exercised: int | None = None
    exercise_price: float | None = None

    def dedup_key(self) -> tuple:
        """Stable identity used for de-duplication across overlapping chunks."""
        return (
            self.event_type,
            self.date,
            self.shares,
            self.ratio,
            self.new_capital,
            self.shares_offered,
        )


class Citation(BaseModel):
    """Page-level provenance for a claim or retrieved event.

    Only built when page_number AND source_snippet are genuinely known —
    page numbers are never guessed.
    """

    event_id: str
    page_number: int
    source_snippet: str
    section_heading: str | None = None
    bbox: list[float] | None = None


class PageText(BaseModel):
    """One page of extracted text, with OCR provenance."""

    page_number: int  # 1-indexed
    text: str
    ocr_used: bool = False
    confidence: float = 1.0


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_file: str
    total_pages: int | None = None
    extraction_date: str | None = None
    capital_events: list[CapitalEvent] = Field(default_factory=list)


class SearchHit(BaseModel):
    event: dict[str, Any]
    score: float
    text: str
    event_id: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: Literal["extractive", "llm"]
    # backward-compatible: full event dicts backing the answer
    citations: list[dict[str, Any]] = Field(default_factory=list)
    # page-level provenance; only events with a known page appear here
    page_citations: list[Citation] = Field(default_factory=list)
    retrieval_strategy: str = "vector"  # "hybrid" | "vector" | "bm25"
    query_time_ms: float = 0.0


def citation_from_hit(hit: SearchHit) -> Citation | None:
    """Build a Citation from a search hit, or None when provenance is absent.

    Never invents a page number: both page_number and source_snippet must
    be present in the stored event metadata.
    """
    page = hit.event.get("page_number")
    snippet = hit.event.get("source_snippet")
    if page is None or snippet is None:
        return None
    return Citation(
        event_id=hit.event_id or f"{hit.event.get('event_type', 'event')}@p{page}",
        page_number=int(page),
        source_snippet=str(snippet),
        bbox=hit.event.get("bbox") if isinstance(hit.event.get("bbox"), list) else None,
    )
