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
# Only retry the (noisier) text strategy when a page strongly looks like a
# capital-structure *table*, not merely a page that mentions shares in prose.
# Broad single-word signals made the text pass shatter prose into fake tables
# (e.g. an IPO-performance page) and emit garbage events; these tight phrases
# fire only on real build-up / allotment-history tables.
FALLBACK_SIGNALS = ("date of allotment", "build-up of", "build up of",
                    "history of equity share capital",
                    "history of the equity share capital",
                    "history of share capital", "history of the share capital",
                    "increased from")


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
_DATE_FORMATS = (
    "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d/%m/%y",
    "%B %d, %Y", "%B %d %Y", "%d %B %Y", "%d %B, %Y",
    "%b %d, %Y", "%b %d %Y", "%d %b %Y", "%d-%b-%Y", "%d-%b-%y",
)
# Magnitude words that may trail a shorthand amount ("Rs. 150 crore").
_MULTIPLIERS = (
    ("crore", 10**7), ("cr", 10**7), ("lakh", 10**5), ("lac", 10**5),
    ("billion", 10**9), ("bn", 10**9), ("million", 10**6), ("mn", 10**6),
)
# Only treat a magnitude word as a multiplier when the number itself is a
# shorthand (below 1 lakh). This protects the common DRHP pattern where the
# full figure AND a spelled-out word coexist:
# "1,50,00,00,000 (Rupees One Hundred Fifty Crore)" must stay 1.5e9, not *1e7.
_MULTIPLIER_CEILING = 100_000
# Footnote / reference markers that trail real cell values in filings.
_FOOTNOTE_RE = re.compile(r"[\*#^~†‡†‡]+|\(\d+\)|\[\d+\]")


def parse_amount(cell: str | None) -> Optional[float]:
    """Parse an Indian/standard money or number cell to a float, or None.

    Handles thousands/lakh grouping ("2,50,000" -> 250000.0), currency
    decoration ("Rs."/"INR"/"₹"), and shorthand magnitude words
    ("Rs. 150 crore" -> 1.5e9, "10 lakh" -> 1e6). Returns None when no number
    is found (e.g. "Nil", "N/A", "—").
    """
    if cell is None:
        return None
    text = str(cell)
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        value = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    if value < _MULTIPLIER_CEILING:
        low = text.lower()
        for word, factor in _MULTIPLIERS:
            if re.search(rf"\b{word}\b", low):
                return value * factor
    return value


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
    # Collapse internal whitespace so wrapped cells ("January 31,\n2019")
    # match the date formats below instead of falling through to raw, and drop
    # trailing footnote markers ("December 19, 2023*" / "...(1)") that filings
    # attach to dates.
    raw = " ".join(str(cell).split())
    raw = _FOOTNOTE_RE.sub("", raw).strip().rstrip(",").strip()
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


def _squash(text: str) -> str:
    """Lower-case and drop punctuation/spacing so header variants converge.

    "No. of Equity Shares" / "Number  of equity-shares" both squash to
    "noofequityshares", letting one needle match many real-filing spellings
    without widening into unrelated columns.
    """
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _find_col(headers: list[str], *needles: str, exclude: tuple = ()) -> Optional[int]:
    """Index of the first header matching any needle (punctuation-insensitive).

    Matching is done on the squashed form (letters+digits only), so "no." vs
    "no", extra spaces, and hyphenation don't cause misses. Needles stay
    specific enough that the precision guards (allotment date, nature of
    transaction) are unaffected. ``exclude`` skips known columns — e.g. the
    share-count search excludes the date column, since "date of allotment of
    equity shares" also contains "shares".
    """
    squashed = [_squash(h) for h in headers]
    keys = [_squash(n) for n in needles]
    for i, h in enumerate(squashed):
        if i in exclude:
            continue
        if any(k and k in h for k in keys):
            return i
    return None


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# A grouped ("10,000") or long (>=4 digit) number — the signature of a value
# row, used to tell a real data row from a wrapped header-continuation row.
_DATA_NUM_RE = re.compile(r"\d{1,3}(?:,\d{2,3})+|\d{4,}")


def _row_is_data(cells: list) -> bool:
    """True if a row looks like table data (carries a real date or a
    grouped/large number) rather than a continuation of a wrapped header."""
    for c in cells:
        s = " ".join(str(c or "").split())
        if not s:
            continue
        parsed = parse_date(s)
        if parsed and _ISO_RE.match(parsed):
            return True
        if _DATA_NUM_RE.search(s):
            return True
    return False


def _merge_multiline_header(data: list) -> tuple[list[str], list]:
    """Combine leading header-continuation rows into one header row.

    DRHP tables routinely wrap a header like "Date of Allotment" across two or
    three physical rows; pdfplumber then keeps only "date of" in the header and
    pushes the rest into the first "data" rows, so the column never matches its
    needle. We absorb up to three leading non-data rows into the header
    (cell-wise) until the first real data row — one carrying a date or grouped
    number. Tables with a normal single-row header are unaffected.
    """
    if not data:
        return [], []
    header_rows = [data[0]]
    idx = 1
    while idx < len(data) and idx <= 3 and not _row_is_data(data[idx]):
        header_rows.append(data[idx])
        idx += 1
    ncols = max(len(r) for r in header_rows)
    headers = [
        _norm(" ".join(str(hr[c]) for hr in header_rows if c < len(hr) and hr[c]))
        for c in range(ncols)
    ]
    return headers, data[idx:]


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

    # Strict allotment date: a bare "date" needle also matched "date of
    # acquisition" / "date of transfer" tables, mis-typing secondary share
    # acquisitions as primary allotments. The allotment branch uses this
    # strict column; bonus / authorised keep the looser date_i for their
    # own date field (they're gated by ratio / increased-from-to anyway).
    allot_date_i = _find_col(headers, "date of allotment", "allotment date")
    date_i = _find_col(headers, "date of allotment", "allotment date",
                       "date of resolution", "record date", "date")
    # Share count must not resolve to the date column ("date of allotment of
    # equity shares" contains "shares") nor to a running "cumulative" column.
    _share_exclude = tuple(
        i for i, h in enumerate(headers)
        if (allot_date_i is not None and i == allot_date_i) or "cumulative" in h
    )
    shares_i = _find_col(headers, "no. of shares", "number of shares", "no of shares",
                         "no. of equity", "number of equity", "shares allotted",
                         "shares issued", "shares offered", "shares",
                         exclude=_share_exclude)
    # Mixed allotment/transfer tables (e.g. "date of allotment/transfer" with a
    # "nature of transaction" = Allotment | Transfer column) list secondary
    # share transfers alongside primary allotments. Only primary allotments are
    # company issuances; when this column exists we keep only those rows.
    nature_txn_i = _find_col(headers, "nature of transaction")
    # Equity-only guard: a "number of preference shares" / pure-CCPS / warrant
    # table is a different instrument. Skip it only when the share column names
    # a non-equity instrument AND does not also mention equity (so a mixed
    # "equity shares / CCPS transacted" column is still kept).
    _share_hdr = headers[shares_i] if shares_i is not None else ""
    is_non_equity = "equity" not in _share_hdr and any(
        w in _share_hdr for w in ("preference", "ccps", "ccp", "warrant", "debenture"))
    face_i = _find_col(headers, "face value")
    price_i = _find_col(headers, "issue price", "price per", "offer price")
    consid_i = _find_col(headers, "consideration", "nature of", "nature")
    ratio_i = _find_col(headers, "ratio")
    record_i = _find_col(headers, "record date")
    # Require the explicit "increased from/to" phrasing. Bare "from"/"to" as
    # substrings matched prose fragments ("...preceding year from") and turned
    # non-tables into bogus authorised-capital events.
    from_i = _find_col(headers, "increased from")
    to_i = _find_col(headers, "increased to")
    reso_i = _find_col(headers, "type of", "resolution type", "nature of resolution")

    def cell(row: dict, i: Optional[int]) -> Optional[str]:
        if i is None or i >= len(headers):
            return None
        return row.get(headers[i])

    # Observability: which signature columns this table exposed. A reviewer
    # tailing the logs at DEBUG can see why a table did or didn't match.
    logger.debug(
        "table p%d cols: allot_date=%s shares=%s ratio=%s incr=%s/%s nature_txn=%s",
        chunk.page_num, allot_date_i, shares_i, ratio_i, from_i, to_i, nature_txn_i,
    )

    events: list[dict] = []
    for row in chunk.rows:
        row_text = " | ".join(str(v) for v in row.values() if v)

        # authorised capital change — needs an explicit from→to pair.
        if from_i is not None and to_i is not None:
            old_cap = parse_int(cell(row, from_i))
            new_cap = parse_int(cell(row, to_i))
            # Authorised capital is always large (lakhs/crores). A tiny value
            # means we matched stray digits in prose, not a real capital table.
            if old_cap is not None and new_cap is not None and new_cap >= 100_000:
                ev = {
                    "event_type": "authorised_capital_change",
                    "date": parse_date(cell(row, date_i) or cell(row, reso_i)),
                    "old_capital": old_cap,
                    "new_capital": new_cap,
                    "resolution_type": _norm(cell(row, reso_i)) or None,
                }
                events.append({**ev, **_provenance(chunk, row_text)})
                continue

        # bonus / rights issue — a ratio column is the signature, but it must
        # sit in a capital-event table (carrying a record date or allotment
        # date), never a generic financial-ratios table (debt-equity, current
        # ratio, etc.). That date requirement is the precision guard.
        if ratio_i is not None and (record_i is not None or allot_date_i is not None):
            ratio_raw = cell(row, ratio_i)
            if ratio_raw and re.search(r"\d[\d,]*\s*:\s*\d[\d,]*", str(ratio_raw)):
                # Bonus shares are free; rights are priced. Disambiguate on an
                # explicit "rights"/"bonus" signal in the section or headers.
                ctx = _norm(chunk.source_section + " " + " ".join(headers))
                is_rights = "rights" in ctx and "bonus" not in ctx
                ev = {
                    "event_type": "rights_issue" if is_rights else "bonus_issue",
                    "date": parse_date(cell(row, record_i) or cell(row, date_i)),
                    "ratio": re.sub(r"\s+", "", str(ratio_raw)),
                }
                if is_rights:
                    ev["price"] = parse_amount(cell(row, price_i))
                    ev["shares_offered"] = parse_int(cell(row, shares_i))
                else:
                    ev["shares_issued"] = parse_int(cell(row, shares_i))
                events.append({**ev, **_provenance(chunk, row_text)})
                continue

        # allotment — a "date of allotment" column + an equity share count is
        # the signature (strict date column keeps acquisitions/transfers out;
        # is_non_equity keeps preference/CCPS tables out).
        if allot_date_i is not None and shares_i is not None and not is_non_equity:
            # In a mixed table, skip rows explicitly marked as a transfer /
            # acquisition rather than an allotment.
            if nature_txn_i is not None and "allotment" not in _norm(cell(row, nature_txn_i)):
                continue
            shares = parse_int(cell(row, shares_i))
            date = parse_date(cell(row, allot_date_i))
            # Require a *validly parsed* date (ISO), not a raw fallback — a
            # wrapped/split date like "December 23" (no year) is not trustworthy.
            if shares is not None and date is not None and _ISO_RE.match(date):
                ev = {
                    "event_type": "allotment",
                    "date": date,
                    "shares": shares,
                    "face_value": parse_amount(cell(row, face_i)),
                    "issue_price": parse_amount(cell(row, price_i)),
                    "consideration": _norm(cell(row, consid_i)) or None,
                }
                events.append({**ev, **_provenance(chunk, row_text)})

    # Only surface tables that actually produced events; a long filing has
    # hundreds of non-capital tables and logging each at INFO is just noise.
    if events:
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
                strategy = "lines"
                tables = page.find_tables(table_settings=self.table_settings)
                if not tables:
                    text = (page.extract_text() or "").lower()
                    if any(sig in text for sig in FALLBACK_SIGNALS):
                        tables = page.find_tables(table_settings=TEXT_SETTINGS)
                        strategy = "text-fallback"
                if not tables:
                    continue
                logger.debug("p%d: %d table(s) via %s strategy",
                             page.page_number, len(tables), strategy)
                # Only infer section headings on pages that actually have a
                # table — the char-level scan is wasted on the other ~half.
                sections = self._infer_sections(page)
                for tbl in tables:
                    data = tbl.extract()
                    if not data or len(data) < 2:
                        continue
                    # Fold any wrapped/multi-line header rows into one header so
                    # split column names ("date of" + "allotment") still match.
                    headers, body = _merge_multiline_header(data)
                    rows = [
                        {headers[i]: (cell or "").strip()
                         for i, cell in enumerate(r) if i < len(headers)}
                        for r in body
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
        """Convenience: extract tables and convert each to events.

        De-duplicated within the filing, since the same allotment is often
        restated across tables (e.g. an "equity share capital history" and a
        separate "allotments for cash" table list identical date+share rows).
        """
        events: list[dict] = []
        for chunk in self.extract(pdf_path):
            events.extend(table_to_events(chunk))
        return merge_with_dedup(events, [])