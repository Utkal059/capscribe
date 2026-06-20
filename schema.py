"""Shared data models for CapScribe.

A single permissive `CapitalEvent` model covers all four event families
(allotment, bonus_issue, rights_issue, authorised_capital_change) because
each family carries a different field set. `extra="allow"` keeps any
unmodelled fields rather than dropping them, while the typed fields below
give validation and editor support for the common ones.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "allotment",
    "bonus_issue",
    "rights_issue",
    "authorised_capital_change",
]


class CapitalEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_type: EventType
    date: str | None = None

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


class AskResponse(BaseModel):
    question: str
    answer: str
    mode: Literal["extractive", "llm"]
    citations: list[dict[str, Any]] = Field(default_factory=list)
