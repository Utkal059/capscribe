"""
capscribe — postprocess.py
Deduplication, Pydantic validation, and CSV export.

Fixes applied (vs original):
  - Deduplication key now matches extractor.py's actual field names
    (shares / face_value / issue_price, not amount / securities_count)
  - Pydantic models validate every event; malformed events are logged
    to invalid_events.json instead of silently corrupting the output
  - Dedup key uses a full-dict hash for robustness (not 4 hand-picked fields)
  - CLI now writes the cleaned JSON back alongside the CSV
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("[warn] pydantic not installed — schema validation skipped. Run: pip install pydantic")


# ── Pydantic models ────────────────────────────────────────────────────────────

if PYDANTIC_AVAILABLE:
    class AllotmentEvent(BaseModel):
        event_type: str
        date: Optional[str] = None
        shares: Optional[int] = None
        face_value: Optional[float] = None
        issue_price: Optional[float] = None
        consideration: Optional[str] = None
        allottee_category: Optional[str] = None

    class BonusIssueEvent(BaseModel):
        event_type: str
        date: Optional[str] = None
        ratio: Optional[str] = None
        shares_issued: Optional[int] = None
        pre_issue_capital: Optional[int] = None
        post_issue_capital: Optional[int] = None

    class RightsIssueEvent(BaseModel):
        event_type: str
        date: Optional[str] = None
        ratio: Optional[str] = None
        price: Optional[float] = None
        shares_offered: Optional[int] = None

    class AuthorisedCapitalChange(BaseModel):
        event_type: str
        date: Optional[str] = None
        old_capital: Optional[float] = None
        new_capital: Optional[float] = None
        resolution_type: Optional[str] = None

    _SCHEMA_MAP = {
        "allotment": AllotmentEvent,
        "bonus_issue": BonusIssueEvent,
        "rights_issue": RightsIssueEvent,
        "authorised_capital_change": AuthorisedCapitalChange,
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_event(event: dict) -> tuple[bool, Any]:
    """
    Returns (is_valid, parsed_model_or_raw_dict).
    Falls back to accepting the raw dict when Pydantic isn't installed.
    """
    if not PYDANTIC_AVAILABLE:
        return True, event

    etype = event.get("event_type", "").lower()
    model_cls = _SCHEMA_MAP.get(etype)
    if model_cls is None:
        # Unknown event type — accept as-is with a warning
        return True, event

    try:
        parsed = model_cls.model_validate(event)
        return True, parsed.model_dump(exclude_none=False)
    except ValidationError as exc:
        return False, str(exc)


# ── Deduplication ──────────────────────────────────────────────────────────────

def _event_hash(event: dict) -> str:
    """
    Stable hash of the normalised event dict.
    Using the full dict is more robust than hand-picking 4 fields.
    """
    # Sort keys for stability, lowercase string values
    normalised = {
        k: (v.lower().strip() if isinstance(v, str) else v)
        for k, v in sorted(event.items())
    }
    blob = json.dumps(normalised, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def deduplicate(events: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for e in events:
        h = _event_hash(e)
        if h not in seen:
            seen.add(h)
            out.append(e)
    return out


# ── CSV export ─────────────────────────────────────────────────────────────────

# All fields that might appear across event types
_CSV_FIELDS = [
    "event_type",
    "date",
    "shares",
    "shares_issued",
    "shares_offered",
    "face_value",
    "issue_price",
    "price",
    "consideration",
    "allottee_category",
    "ratio",
    "pre_issue_capital",
    "post_issue_capital",
    "old_capital",
    "new_capital",
    "resolution_type",
]


def to_csv(events: list[dict], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)
    print(f"CSV saved to {out_path}  ({len(events)} rows)")


# ── Main ───────────────────────────────────────────────────────────────────────

def postprocess(json_path: Path) -> None:
    raw = json.loads(json_path.read_text(encoding="utf-8"))

    # Accept both top-level list and {"capital_events": [...]} wrapper
    if isinstance(raw, dict):
        events = raw.get("capital_events", [])
        meta = {k: v for k, v in raw.items() if k != "capital_events"}
    else:
        events = raw
        meta = {}

    print(f"Loaded {len(events)} raw events from {json_path.name}")

    # ── Validate ──
    valid_events: list[dict] = []
    invalid_events: list[dict] = []

    for e in events:
        ok, result = validate_event(e)
        if ok:
            valid_events.append(result)
        else:
            invalid_events.append({"raw": e, "error": result})

    if invalid_events:
        bad_path = json_path.parent / f"{json_path.stem}_invalid.json"
        bad_path.write_text(json.dumps(invalid_events, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[warn] {len(invalid_events)} invalid events logged to {bad_path.name}")

    # ── Deduplicate ──
    deduped = deduplicate(valid_events)
    print(f"After deduplication: {len(deduped)} unique events  (removed {len(valid_events) - len(deduped)} duplicates)")

    # ── Write cleaned JSON ──
    clean_json_path = json_path.parent / f"{json_path.stem}_clean.json"
    output = {**meta, "capital_events": deduped}
    clean_json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Clean JSON saved to {clean_json_path.name}")

    # ── Write CSV ──
    csv_path = json_path.with_suffix(".csv")
    to_csv(deduped, csv_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python postprocess.py <path-to-extracted.json>")
        sys.exit(1)

    postprocess(Path(sys.argv[1]))
