"""Structured table extraction for financial filings.

Financial filings are mostly tables, but the text/LLM pass treats them as
prose and loses their structure. This module pulls tables out directly with
``pdfplumber``, infers the section heading each table sits under, and
pattern-matches the common capital-event tables (allotments, bonus issues,
authorised-capital changes) straight into the event schema the rest of the
pipeline uses — carrying table-level provenance (page + bbox + section) so
every field is citable.

Design notes:
  - ``TableExtractor.extract`` needs a real PDF and ``pdfplumber``.
  - ``table_to_events`` is a *pure* function over a :class:`TableChunk`, so
    the detection/parsing logic is fully unit-testable with no PDF.
  - Table-extracted events are higher structural fidelity than LLM-extracted
    ones, so ``merge_with_dedup`` prefers the table version on a fuzzy match.
  - Real DRHP tables are inconsistently ruled. We try the ruled-line strategy
    first and, on capital-relevant pages where it finds nothing, fall back to
    the text strategy so borderless / whitespace-aligned tables still parse.
"""
from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pdfplumber

logger = logging.getLogger("capscribe.tables")

# Ruled tables (clean grids) and borderless tables (whitespace-aligned) need
# different pdfplumber strategies. Try lines first, then fall back to text on
# pages that look capital-relevant (keeps the noisier text pass targeted).
LINES_SETTINGS = {"vertical_strategy": "lines", "horizontal_strategy": "lines", "snap_tolerance": 4, "join_tolerance": 4}
TEXT_SETTINGS = {"vertical_strategy": "text", "horizontal_strategy": "text", "snap_tolerance": 4, "join_tolerance": 4}
TABLE_SETTINGS = LINES_SETTINGS  # back-compat: TableExtractor default + tests
CAPITAL_SIGNALS = ("date of allotment", "equity share capital", "bonus", "rights issue",
                   "authorised capital", "authorized capital", "allotment", "face value")


@dataclass
class TableChunk:
    """One table lifted from a page, with structural provenance.

    Attributes:
        page_num: 1-indexed source page.
        bbox: (x0, top, x1, bottom) bounding box of the table on the page.
        headers: normalised header cell strings (lower-cased, ws-collapsed).
        rows: list of ``{header: cell}`` dicts, one per body row.
        raw_text: the table flattened to text (used as a citation snippet).
        source_section: nearest preceding heading on the page, or "".
    """

    page_num: int
    bbox: tuple
    headers: list[str]
    rows: list[dict] = field(default_factory=list)
    raw_text: str = ""
    source_section: str = ""


# ── value parsing ──────────────────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%B %d, %Y", "%d %B %Y", "%b %d, %Y")


def parse_amount(cell: str | None) -> Optional[float]:
    """Parse an Indian-formatted money/number cell to a float, or None.

    Handles thousands/lakh grouping ("2,50,000" -> 250000.0) and currency
    decoration ("Rs. 154.20" -> 154.2). Returns None when no number is found.
    """
    if cell is None:
        return None
    m = _NUM_RE.search(str(cell))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_int(cell: str | None) -> Optional[int]:
    """Parse a cell to an int (via :func:`parse_amount`), or None."""
    val = parse_amount(cell)
    return int(val) if val is not None else None


def parse_date(cell: str | None) -> Optional[str]:
    """Parse a date cell to ISO ``YYYY-MM-DD``; fall back to the raw string.

    Returns None for empty cells so callers can distinguish "no date" from
    "unparseable date".
    """
    if cell is None:
        return None
    raw = str(cell).strip()
    if not raw:
        return None
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def _norm(text: str | None) -> str:
    return " ".join(str(text or "").split()).lower()


def _find_col(headers: list[str], *needles: str) -> Optional[int]:
    """Index of the first header containing any needle substring, else None."""
    for i, h in enumerate(headers):
        if any(n in h for n in needles):
            return i
    return None


# ── table → events ──────────────────────────────────────────────────────────────

def _provenance(chunk: TableChunk, row_text: str) -> dict:
    """Common flat + structured provenance for a table-sourced event."""
    bbox = list(chunk.bbox) if chunk.bbox else None
    return {
        "page_number": chunk.page_num,
        "page": chunk.page_num,  # alias for UIs that read `page`
        "source_snippet": row_text[:200] if row_text else chunk.raw_text[:200],
        "bbox": bbox,
        "confidence": 0.95,  # structural extraction, but headings can mislead
        "source_provenance": {
            "page": chunk.page_num,
            "section": chunk.source_section or None,
            "bbox": bbox,
            "extraction_method": "table",
            "confidence": 0.95,
        },
    }


def table_to_events(chunk: TableChunk) -> list[dict]:
    """Pattern-match a table into capital events (schema-shaped dicts).

    Detects three table families by their column signatures:
      - allotment: Date + No. of Shares (+ Face Value / Issue Price / Consideration)
      - bonus_issue: Ratio (+ Record Date)
      - authorised_capital_change: Increased From + Increased To (+ Resolution)

    Returns an empty list when the table matches no known family, so callers
    can pass every table through cheaply.
    """
    headers = chunk.headers
    if not headers or not chunk.rows:
        return []

    date_i = _find_col(headers, "date of allotment", "date")
    shares_i = _find_col(headers, "no. of shares", "number of shares", "no of shares",
                         "no. of equity", "number of equity", "shares allotted", "shares")
    face_i = _find_col(headers, "face value")
    price_i = _find_col(headers, "issue price", "price per")
    consid_i = _find_col(headers, "consideration", "nature of", "nature")
    ratio_i = _find_col(headers, "ratio")
    record_i = _find_col(headers, "record date")
    from_i = _find_col(headers, "increased from", "from")
    to_i = _find_col(headers, "increased to", "to")
    reso_i = _find_col(headers, "type of", "resolution type", "nature of resolution")

    def cell(row: dict, i: Optional[int]) -> Optional[str]:
        if i is None or i >= len(headers):
            return None
        return row.get(headers[i])

    events: list[dict] = []
    for row in chunk.rows:
        row_text = " | ".join(str(v) for v in row.values() if v)

        # authorised capital change — needs an explicit from→to pair.
        if from_i is not None and to_i is not None:
            ev = {
                "event_type": "authorised_capital_change",
                "date": parse_date(cell(row, date_i) or cell(row, reso_i)),
                "old_capital": parse_int(cell(row, from_i)),
                "new_capital": parse_int(cell(row, to_i)),
                "resolution_type": _norm(cell(row, reso_i)) or None,
            }
            if ev["new_capital"] is not None:
                events.append({**ev, **_provenance(chunk, row_text)})
                continue

        # bonus issue — a ratio column is the signature.
        if ratio_i is not None:
            ratio_raw = cell(row, ratio_i)
            if ratio_raw and re.search(r"\d+\s*:\s*\d+", str(ratio_raw)):
                ev = {
                    "event_type": "bonus_issue",
                    "date": parse_date(cell(row, record_i) or cell(row, date_i)),
                    "ratio": re.sub(r"\s+", "", str(ratio_raw)),
                    "shares_issued": parse_int(cell(row, shares_i)),
                }
                events.append({**ev, **_provenance(chunk, row_text)})
                continue

        # allotment — date + a share count is the signature.
        if date_i is not None and shares_i is not None:
            shares = parse_int(cell(row, shares_i))
            date = parse_date(cell(row, date_i))
            if shares is not None and date is not None:
                ev = {
                    "event_type": "allotment",
                    "date": date,
                    "shares": shares,
                    "face_value": parse_amount(cell(row, face_i)),
                    "issue_price": parse_amount(cell(row, price_i)),
                    "consideration": _norm(cell(row, consid_i)) or None,
                }
                events.append({**ev, **_provenance(chunk, row_text)})

    logger.info("table p%d (%s): %d events", chunk.page_num,
                chunk.source_section or "—", len(events))
    return events


# ── dedup against the LLM pass ───────────────────────────────────────────────────

def _dedup_key(ev: dict) -> tuple:
    return (ev.get("event_type"), ev.get("date"), ev.get("shares"), ev.get("ratio"),
            ev.get("new_capital"))


def merge_with_dedup(table_events: list[dict], llm_events: list[dict]) -> list[dict]:
    """Merge table- and LLM-extracted events, preferring table on a match.

    Table extraction has higher structural fidelity, so when an LLM event
    fuzzy-matches a table event on ``(event_type, date, shares|ratio|new_capital)``
    the table version wins. Non-matching LLM events are kept.
    """
    merged: dict[tuple, dict] = {}
    for ev in llm_events:
        merged[_dedup_key(ev)] = ev
    for ev in table_events:  # table overwrites the LLM version on collision
        merged[_dedup_key(ev)] = ev
    return list(merged.values())


# ── PDF-backed extraction (needs pdfplumber + a real PDF) ────────────────────────

class TableExtractor:
    """Extract structured tables (and their sections) from a PDF."""

    def __init__(self, table_settings: Optional[dict] = None) -> None:
        self.table_settings = table_settings or TABLE_SETTINGS

    def _infer_sections(self, page: "pdfplumber.page.Page") -> list[tuple[float, str]]:
        """Heading candidates as (top_y, text) for lines with above-median font size."""
        chars = page.chars or []
        if not chars:
            return []
        median_size = statistics.median(c.get("size", 0) for c in chars)
        lines: dict[int, dict] = {}
        for c in chars:
            key = round(c.get("top", 0))
            line = lines.setdefault(key, {"top": c.get("top", 0), "text": "", "sizes": []})
            line["text"] += c.get("text", "")
            line["sizes"].append(c.get("size", 0))
        headings = []
        for line in lines.values():
            if line["sizes"] and statistics.mean(line["sizes"]) > median_size * 1.1:
                text = " ".join(line["text"].split())
                if text:
                    headings.append((line["top"], text))
        return sorted(headings)

    def _section_for(self, sections: list[tuple[float, str]], table_top: float) -> str:
        """Nearest heading appearing above the table."""
        above = [text for top, text in sections if top <= table_top]
        return above[-1] if above else ""

    def extract(self, pdf_path: str | Path) -> list[TableChunk]:
        """Return every table on every page as a :class:`TableChunk`.

        Tries the ruled-line strategy first; on pages that find no ruled table
        but contain capital-event signals, retries with the text strategy so
        borderless / whitespace-aligned DRHP tables are still captured.
        """
        import pdfplumber

        chunks: list[TableChunk] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                sections = self._infer_sections(page)
                tables = page.find_tables(table_settings=self.table_settings)
                if not tables:
                    text = (page.extract_text() or "").lower()
                    if any(sig in text for sig in CAPITAL_SIGNALS):
                        tables = page.find_tables(table_settings=TEXT_SETTINGS)
                for tbl in tables:
                    data = tbl.extract()
                    if not data or len(data) < 2:
                        continue
                    headers = [_norm(h) for h in data[0]]
                    rows = [
                        {headers[i]: (cell or "").strip()
                         for i, cell in enumerate(r) if i < len(headers)}
                        for r in data[1:]
                    ]
                    raw_text = "\n".join(
                        " | ".join((c or "") for c in r) for r in data
                    )
                    chunks.append(
                        TableChunk(
                            page_num=page.page_number,
                            bbox=tuple(round(float(b), 2) for b in tbl.bbox),
                            headers=headers,
                            rows=rows,
                            raw_text=raw_text,
                            source_section=self._section_for(sections, tbl.bbox[1]),
                        )
                    )
        logger.info("extracted %d tables from %s", len(chunks), pdf_path)
        return chunks

    def extract_events(self, pdf_path: str | Path) -> list[dict]:
        """Convenience: extract tables and convert each to events."""
        events: list[dict] = []
        for chunk in self.extract(pdf_path):
            events.extend(table_to_events(chunk))
        return events