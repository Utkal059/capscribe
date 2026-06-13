"""Source-backed capital-history report generation.

The JD's agentic workflow ends in *report generation*: turning the verified,
cited event set into an analyst-ready brief. This module is the final stage
after ``retrieval -> reasoning -> verification``.

Design (mirrors the rest of the system — "simple, observable systems over
clever but fragile ones"):

  - The **facts are computed in Python** — a chronological timeline,
    per-type rollups, headline capital metrics, and the verification
    findings — so the figures are deterministic and never hallucinated.
  - Every claim carries an **inline page citation** ``(p. N)`` built only
    from events that genuinely record a page + snippet; pages are never
    invented (same rule as ``schema.citation_from_hit``).
  - ``mode="extractive"`` (default) is free and offline. ``mode="llm"``
    asks Claude to write a short executive summary **from the computed
    skeleton only**, so the prose cannot introduce a number the events
    don't support.

The public entry point :func:`generate_report` takes a list of event dicts
(as stored in the vector collection) and returns a :class:`CapitalReport`
with both structured fields and a rendered Markdown brief.
"""
from __future__ import annotations

import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from config import settings
from schema import Citation
from verification import verify_report

logger = logging.getLogger("capscribe.report")


# ── models ────────────────────────────────────────────────────────────────────

class TimelineEntry(BaseModel):
    date: str | None = None
    event_type: str
    headline: str
    page_number: int | None = None
    source_snippet: str | None = None


class CapitalReport(BaseModel):
    """An analyst-ready capital-history brief over an extracted filing."""

    title: str
    mode: Literal["extractive", "llm"] = "extractive"
    event_count: int = 0
    by_type: dict[str, int] = Field(default_factory=dict)
    date_range: list[str | None] = Field(default_factory=lambda: [None, None])
    metrics: dict[str, Any] = Field(default_factory=dict)
    timeline: list[TimelineEntry] = Field(default_factory=list)
    verification: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    narrative: str = ""        # short prose summary (llm mode) or "" (extractive)
    markdown: str = ""         # full rendered brief


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_inr(amount: float | int | None) -> str:
    """Render a rupee figure with crore/lakh scale for readability."""
    if amount is None:
        return "—"
    a = float(amount)
    if a >= 1e7:
        return f"Rs. {a / 1e7:.2f} Cr"
    if a >= 1e5:
        return f"Rs. {a / 1e5:.2f} L"
    return f"Rs. {a:,.0f}"


def _cite(ev: dict) -> str:
    """Inline ``(p. N)`` marker when a page is known, else ""."""
    page = ev.get("page_number")
    return f" (p. {page})" if page is not None else ""


def _headline(ev: dict) -> str:
    """One-line human summary of an event for the timeline."""
    et = ev.get("event_type", "event")
    if et == "allotment":
        who = ev.get("allottee_category")
        bit = f" to {who}" if who else ""
        price = ev.get("issue_price")
        at = f" at Rs. {price}" if price else ""
        return f"Allotted {ev.get('shares', '?'):,} shares{bit}{at}".replace("?,", "?")
    if et == "bonus_issue":
        return f"Bonus issue {ev.get('ratio', '?')}"
    if et == "rights_issue":
        return f"Rights issue {ev.get('ratio', '?')} at Rs. {ev.get('price', '?')}"
    if et == "authorised_capital_change":
        return (f"Authorised capital raised {_fmt_inr(ev.get('old_capital'))} "
                f"→ {_fmt_inr(ev.get('new_capital'))}")
    if et == "dividend_declaration":
        return f"Dividend Rs. {ev.get('amount_per_share', '?')}/share"
    if et == "share_repurchase":
        n = ev.get("shares_bought_back")
        return f"Buyback of {n:,} shares" if isinstance(n, int) else "Share repurchase"
    if et == "warrant_exercise":
        n = ev.get("warrants_exercised")
        return f"{n:,} warrants exercised" if isinstance(n, int) else "Warrant exercise"
    return et.replace("_", " ")


def _safe_int_sum(events: list[dict], etype: str, field: str) -> int:
    return sum(int(e[field]) for e in events
              if e.get("event_type") == etype and isinstance(e.get(field), (int, float)))


def _metrics(events: list[dict]) -> dict[str, Any]:
    """Headline capital metrics rolled up across the event set."""
    acc = sorted(
        (e for e in events if e.get("event_type") == "authorised_capital_change"),
        key=lambda e: (e.get("date") or ""),
    )
    latest_capital = acc[-1].get("new_capital") if acc else None
    return {
        "total_shares_allotted": _safe_int_sum(events, "allotment", "shares"),
        "allotment_events": sum(1 for e in events if e.get("event_type") == "allotment"),
        "bonus_issues": sum(1 for e in events if e.get("event_type") == "bonus_issue"),
        "rights_issues": sum(1 for e in events if e.get("event_type") == "rights_issue"),
        "latest_authorised_capital": latest_capital,
        "latest_authorised_capital_fmt": _fmt_inr(latest_capital),
        "total_dividend_outflow": _safe_int_sum(events, "dividend_declaration", "total_outflow"),
        "total_shares_bought_back": _safe_int_sum(events, "share_repurchase", "shares_bought_back"),
        "total_warrants_exercised": _safe_int_sum(events, "warrant_exercise", "warrants_exercised"),
    }


def _timeline(events: list[dict]) -> list[TimelineEntry]:
    dated = sorted(events, key=lambda e: (e.get("date") or "9999-99-99"))
    return [
        TimelineEntry(
            date=e.get("date"),
            event_type=e.get("event_type", "event"),
            headline=_headline(e),
            page_number=e.get("page_number"),
            source_snippet=e.get("source_snippet"),
        )
        for e in dated
    ]


def _citations(events: list[dict]) -> list[Citation]:
    """Page-backed citations — only for events with a real page + snippet."""
    out: list[Citation] = []
    for i, e in enumerate(events):
        page, snippet = e.get("page_number"), e.get("source_snippet")
        if page is None or snippet is None:
            continue
        section = (e.get("source_provenance") or {}).get("section")
        out.append(Citation(
            event_id=str(e.get("event_id") or f"{e.get('event_type', 'event')}@p{page}"),
            page_number=int(page),
            source_snippet=str(snippet),
            section_heading=section,
        ))
    return out


def _render_markdown(report: CapitalReport) -> str:
    m = report.metrics
    lines = [
        f"# {report.title}",
        "",
        f"*{report.event_count} capital events"
        + (f" spanning {report.date_range[0]} → {report.date_range[1]}*"
           if report.date_range[0] else "*"),
        "",
        "## Snapshot",
        "",
        f"- Latest authorised capital: **{m.get('latest_authorised_capital_fmt', '—')}**",
        f"- Total shares allotted: **{m.get('total_shares_allotted', 0):,}** "
        f"across {m.get('allotment_events', 0)} allotments",
        f"- Bonus issues: **{m.get('bonus_issues', 0)}**  ·  "
        f"Rights issues: **{m.get('rights_issues', 0)}**",
        f"- Total dividend outflow: **{_fmt_inr(m.get('total_dividend_outflow') or None)}**",
        f"- Shares bought back: **{m.get('total_shares_bought_back', 0):,}**  ·  "
        f"Warrants exercised: **{m.get('total_warrants_exercised', 0):,}**",
        "",
        "## Capital timeline",
        "",
    ]
    for entry in report.timeline:
        page = f" (p. {entry.page_number})" if entry.page_number is not None else ""
        date = entry.date or "undated"
        lines.append(f"- **{date}** — {entry.headline}{page}")
    lines += ["", "## Verification", ""]
    v = report.verification
    if v.get("consistent", True):
        lines.append(f"- ✅ No contradictions across {v.get('checked', 0)} events "
                     "(timeline / capital-continuity / bonus-arithmetic).")
    else:
        lines.append(f"- ⚠️ **{len(v.get('issues', []))} contradiction(s)** found "
                     f"across {v.get('checked', 0)} events:")
        for issue in v.get("issues", []):
            lines.append(f"    - [{issue.get('check_type')}] {issue.get('description')}")
    if report.narrative:
        lines = [f"# {report.title}", "", "## Executive summary", "",
                 report.narrative, ""] + lines[3:]
    lines += ["", f"*{len(report.citations)} of {report.event_count} events are "
              "page-cited back to the source filing.*"]
    return "\n".join(lines)


# ── LLM narrative (optional) ──────────────────────────────────────────────────

def _llm_narrative(report: CapitalReport) -> str:
    """A short executive paragraph written from the computed skeleton only."""
    from anthropic import Anthropic

    facts = {
        "metrics": report.metrics,
        "timeline": [t.model_dump() for t in report.timeline],
        "verification": report.verification,
    }
    client = Anthropic(api_key=settings.anthropic_api_key or None)
    msg = client.messages.create(
        model=settings.answer_model,
        max_tokens=350,
        system=(
            "You are a capital-markets analyst. Write a 3-4 sentence executive "
            "summary of a company's capital history using ONLY the JSON facts "
            "provided. Never introduce a figure that is not in the facts. Be "
            "precise and neutral; reference dates where relevant."
        ),
        messages=[{"role": "user", "content": str(facts)}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


# ── public API ────────────────────────────────────────────────────────────────

def generate_report(
    events: list[dict],
    mode: Literal["extractive", "llm"] = "extractive",
    title: Optional[str] = None,
) -> CapitalReport:
    """Build a source-backed capital-history report over ``events``.

    Args:
        events: extracted event dicts (as stored in the vector collection).
        mode: ``extractive`` (free, deterministic) or ``llm`` (adds a
            Claude-written executive summary over the same computed facts).
        title: optional report title.

    Returns:
        A :class:`CapitalReport` with structured fields, page citations, the
        verification result, and a rendered Markdown brief.
    """
    dates = sorted(e["date"] for e in events if e.get("date"))
    report = CapitalReport(
        title=title or "Capital History Report",
        mode=mode,
        event_count=len(events),
        by_type=_count_by_type(events),
        date_range=[dates[0] if dates else None, dates[-1] if dates else None],
        metrics=_metrics(events),
        timeline=_timeline(events),
        verification=verify_report(events).model_dump(),
        citations=_citations(events),
    )
    if mode == "llm" and events:
        try:
            report.narrative = _llm_narrative(report)
        except Exception as exc:  # never fail the report on an LLM hiccup
            logger.warning("llm narrative failed, falling back to extractive: %s", exc)
            report.mode = "extractive"
    report.markdown = _render_markdown(report)
    logger.info("report: %d events, %d citations, consistent=%s",
                report.event_count, len(report.citations),
                report.verification.get("consistent"))
    return report


def _count_by_type(events: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for e in events:
        et = e.get("event_type", "unknown")
        out[et] = out.get(et, 0) + 1
    return out
