"""Consistency verification over extracted capital events.

The JD calls out "contradictions" twice: an extraction is only trustworthy
if its events agree with each other. This module runs three deterministic,
offline checks over the event set and reports any contradictions with the
specific events implicated — turning "source-backed" into a property that
can fail loudly rather than a claim.

Checks:
  - timeline    : same-type events on the same date with conflicting figures
  - continuity  : authorised-capital chain where new_capital[t] != old_capital[t+1]
  - arithmetic  : bonus issues where post != pre * (1 + ratio) within tolerance

The checks are pure functions over a list of event dicts, so they unit-test
with no store, no PDF and no network.
"""
from __future__ import annotations

import logging
import re
from typing import Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("capscribe.verify")

BONUS_TOLERANCE = 0.05  # 5% deviation allowed before flagging arithmetic


class Issue(BaseModel):
    event_ids: list[str]
    check_type: Literal["timeline", "capital_continuity", "arithmetic"]
    description: str
    severity: Literal["warning", "error"]


class VerificationResult(BaseModel):
    consistent: bool
    issues: list[Issue] = Field(default_factory=list)
    confidence: float = 1.0


class VerificationReport(BaseModel):
    checked: int
    consistent: bool
    issues: list[Issue] = Field(default_factory=list)
    by_check: dict[str, int] = Field(default_factory=dict)


# ── helpers ──────────────────────────────────────────────────────────────────────

def _eid(ev: dict, fallback: int) -> str:
    return str(ev.get("event_id") or ev.get("id") or
               f"{ev.get('event_type', 'event')}@{ev.get('date') or fallback}")


def ratio_value(ratio: str | None) -> Optional[float]:
    """Convert an "a:b" bonus ratio to the multiplier a/b (e.g. "5:1" -> 5.0)."""
    if not ratio:
        return None
    m = re.match(r"\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$", str(ratio))
    if not m:
        return None
    a, b = float(m.group(1)), float(m.group(2))
    return a / b if b else None


def _sorted_by_date(events: list[dict]) -> list[dict]:
    return sorted(events, key=lambda e: (e.get("date") or ""))


# ── individual checks ─────────────────────────────────────────────────────────────

def check_timeline(events: list[dict]) -> list[Issue]:
    """Flag same-type events sharing a date but disagreeing on key figures."""
    issues: list[Issue] = []
    by_date: dict[tuple, list[tuple[int, dict]]] = {}
    for i, ev in enumerate(events):
        if not ev.get("date"):
            continue
        by_date.setdefault((ev.get("event_type"), ev["date"]), []).append((i, ev))
    for (etype, date), group in by_date.items():
        if len(group) < 2:
            continue
        figures = {(ev.get("shares"), ev.get("ratio"), ev.get("new_capital"))
                   for _, ev in group}
        if len(figures) <= 1:
            continue
        # Same-date allotments at the *same issue price* are almost always one
        # funding round split across allottees (multi-tranche), not a data
        # conflict — downgrade to a warning so analysts aren't told their
        # extraction is broken when it isn't.
        prices = {ev.get("issue_price") for _, ev in group}
        ids = [_eid(ev, i) for i, ev in group]
        if etype == "allotment" and len(prices) == 1 and None not in prices:
            price = next(iter(prices))
            issues.append(Issue(
                event_ids=ids,
                check_type="timeline",
                description=(
                    f"Multi-tranche allotment on {date}: {len(group)} events at the "
                    f"same price (Rs. {price:g}) — likely a single funding round split "
                    f"across allottees."
                ),
                severity="warning",
            ))
        else:
            issues.append(Issue(
                event_ids=ids,
                check_type="timeline",
                description=(
                    f"{len(group)} {etype} events on {date} with conflicting "
                    f"figures (shares/ratio/capital)."
                ),
                severity="error",
            ))
    return issues


def check_capital_continuity(events: list[dict]) -> list[Issue]:
    """Flag breaks in the authorised-capital chain (new[t] != old[t+1])."""
    chain = _sorted_by_date(
        [e for e in events if e.get("event_type") == "authorised_capital_change"]
    )
    issues: list[Issue] = []
    for a, b in zip(chain, chain[1:]):
        new_a, old_b = a.get("new_capital"), b.get("old_capital")
        if new_a is None or old_b is None:
            continue
        if int(new_a) != int(old_b):
            issues.append(Issue(
                event_ids=[_eid(a, 0), _eid(b, 1)],
                check_type="capital_continuity",
                description=(
                    f"authorised capital chain breaks: {a.get('date')} ends at "
                    f"{new_a} but {b.get('date')} starts from {old_b}."
                ),
                severity="error",
            ))
    return issues


def check_bonus_arithmetic(events: list[dict]) -> list[Issue]:
    """Flag bonus issues where post != pre * (1 + ratio) beyond tolerance."""
    issues: list[Issue] = []
    for i, ev in enumerate(events):
        if ev.get("event_type") != "bonus_issue":
            continue
        pre, post = ev.get("pre_issue_capital"), ev.get("post_issue_capital")
        rv = ratio_value(ev.get("ratio"))
        if pre is None or post is None or rv is None or pre == 0:
            continue
        expected = pre * (1 + rv)
        deviation = abs(post - expected) / expected if expected else 1.0
        if deviation > BONUS_TOLERANCE:
            issues.append(Issue(
                event_ids=[_eid(ev, i)],
                check_type="arithmetic",
                description=(
                    f"bonus issue {ev.get('ratio')} on {ev.get('date')}: "
                    f"post {post} != pre {pre} * (1 + {rv:g}) = {expected:g} "
                    f"({deviation:.0%} off)."
                ),
                severity="warning",
            ))
    return issues


# ── public API ────────────────────────────────────────────────────────────────────

def verify_consistency(events: list[dict], event_type: Optional[str] = None) -> VerificationResult:
    """Run all consistency checks over ``events`` (optionally one type only).

    Args:
        events: extracted event dicts.
        event_type: when given, restrict the scope to that event type.

    Returns:
        :class:`VerificationResult` — ``consistent`` is True iff no issues;
        ``confidence`` decays with the number of issues found.
    """
    scope = [e for e in events if e.get("event_type") == event_type] if event_type else events
    issues = (check_timeline(scope)
              + check_capital_continuity(scope)
              + check_bonus_arithmetic(scope))
    confidence = round(max(0.0, 1.0 - 0.2 * len(issues)), 4)
    return VerificationResult(consistent=not issues, issues=issues, confidence=confidence)


def verify_report(events: list[dict]) -> VerificationReport:
    """Full-corpus verification, summarised for the ``/verify`` endpoint."""
    result = verify_consistency(events)
    by_check: dict[str, int] = {}
    for issue in result.issues:
        by_check[issue.check_type] = by_check.get(issue.check_type, 0) + 1
    return VerificationReport(
        checked=len(events),
        consistent=result.consistent,
        issues=result.issues,
        by_check=by_check,
    )


def events_from_store(store) -> list[dict]:
    """Pull all event dicts out of an EventStore / HybridRetriever for checking."""
    target = getattr(store, "store", store)  # unwrap HybridRetriever
    res = target._collection.get()
    metas = res.get("metadatas", []) or []
    ids = res.get("ids", []) or []
    out = []
    for eid, meta in zip(ids, metas):
        ev = dict(meta)
        ev.setdefault("event_id", eid)
        out.append(ev)
    return out
