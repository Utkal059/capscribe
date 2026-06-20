"""Markdown table extraction for filings supplied as ``.md`` instead of PDF.

Some DRHPs reach us already converted to Markdown (e.g. via an upstream
PDF→Markdown pipeline). For those there is nothing to render and nothing to
scan, so the ``pdfplumber`` and ``ocr.py`` routes don't apply. This module
reads the raw Markdown text, parses its GitHub-flavoured tables into the same
:class:`~table_extractor.TableChunk` structure the PDF path builds, and then
reuses :func:`~table_extractor.table_to_events` — so every header-matching
rule, precision guard, and schema mapping is shared with the PDF pipeline
(allotments, bonus/rights issues, authorised-capital changes, …).

Provenance:
  - ATX headings (``#``..``######``) are tracked so each table carries the
    nearest preceding heading as ``source_section`` — exactly like the PDF
    path's inferred sections. That heading drives the bonus-vs-rights split.
  - Page markers left by common converters (``[PAGE 5]``, ``<!-- page 5 -->``,
    a bare ``{5}`` anchor, or ``--- Page 5 ---``) set the event ``page_number``
    so citations still point at a page. With no markers the document is one
    page (``page_number = 1``); page numbers are never invented per-table.

Markdown-sourced events keep ``extraction_method = "table"``: structurally
they *are* tables, just delimited by pipes rather than ruled lines.
"""
from __future__ import annotations

import re
from pathlib import Path

from table_extractor import TableChunk, _norm, merge_with_dedup, table_to_events

# ── line classifiers ─────────────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$")

# Page anchors left behind by PDF→Markdown converters. Best-effort: the first
# non-None group is the page number. Anything unmatched leaves the page as-is.
_PAGE_MARKER_RE = re.compile(
    r"""(?ix)
      \[\s*page\s*(\d+)\s*\]                 # [PAGE 5]
    | <!--\s*page\D*(\d+)\s*-->              # <!-- page 5 --> / <!-- Page: 5 -->
    | ^\s*\{(\d+)\}\s*-*\s*$                 # {5}-----  (marker page anchor)
    | ^\s*-{0,}\s*page\s+(\d+)\s*-{0,}\s*$   # --- Page 5 ---
    """,
)

# A separator cell is dashes with optional alignment colons: ---, :--, --:, :-:
_SEP_CELL_RE = re.compile(r"^:?-+:?$")


def _page_number(line: str) -> int | None:
    """Return the page a marker line declares, or None if it isn't a marker."""
    m = _PAGE_MARKER_RE.search(line)
    if not m:
        return None
    for g in m.groups():
        if g is not None:
            return int(g)
    return None


def _split_row(line: str) -> list[str]:
    """Split a Markdown table row into trimmed cells, honouring escaped pipes."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    parts = re.split(r"(?<!\\)\|", s)
    return [p.replace("\\|", "|").strip() for p in parts]


def _is_separator(line: str) -> bool:
    """True for a table delimiter row (``| --- | :--: |``)."""
    if "-" not in line:  # a delimiter must carry dashes; prose never qualifies
        return False
    cells = [c for c in _split_row(line) if c != ""]
    return bool(cells) and all(_SEP_CELL_RE.fullmatch(c) for c in cells)


# ── parsing ──────────────────────────────────────────────────────────────────

def parse_markdown_tables(md_text: str) -> list[TableChunk]:
    """Parse every GitHub-flavoured table into a :class:`TableChunk`.

    A table is a row line immediately followed by a separator line; the body
    runs until the first line without a pipe (or a blank line). Headers are
    normalised exactly like the PDF path (lower-cased, whitespace-collapsed)
    and each body row is keyed by those headers, so ``table_to_events`` and its
    ``_find_col`` header matching work unchanged.
    """
    lines = md_text.splitlines()
    n = len(lines)
    chunks: list[TableChunk] = []
    current_section = ""
    current_page = 1
    i = 0
    while i < n:
        line = lines[i]

        page = _page_number(line)
        if page is not None:
            current_page = page
            i += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            current_section = heading.group(1).strip()
            i += 1
            continue

        # Table start: a pipe row followed by a separator row.
        if "|" in line and i + 1 < n and _is_separator(lines[i + 1]):
            headers = [_norm(c) for c in _split_row(line)]
            raw_lines = [line, lines[i + 1]]
            rows: list[dict] = []
            j = i + 2
            while j < n and lines[j].strip() and "|" in lines[j] and not _is_separator(lines[j]):
                cells = _split_row(lines[j])
                raw_lines.append(lines[j])
                rows.append({headers[k]: cells[k] for k in range(min(len(headers), len(cells)))})
                j += 1
            chunks.append(
                TableChunk(
                    page_num=current_page,
                    bbox=(),  # markdown has no geometry; provenance bbox stays None
                    headers=headers,
                    rows=rows,
                    raw_text="\n".join(raw_lines),
                    source_section=current_section,
                )
            )
            i = j
            continue

        i += 1
    return chunks


# Spec alias: `parse_md_tables` is the documented public name; the internal
# implementation is `parse_markdown_tables`. Both refer to the same function.
parse_md_tables = parse_markdown_tables


def split_into_pages(md_text: str) -> dict[int, str]:
    """Bucket the document text by page marker (all of it under page 1 when
    no markers are present), mirroring ``ocr.process_document``'s
    ``text_by_page`` so the optional LLM ingest pass can run on Markdown too."""
    pages: dict[int, list[str]] = {}
    current = 1
    for line in md_text.splitlines():
        page = _page_number(line)
        if page is not None:
            current = page
            continue
        pages.setdefault(current, []).append(line)
    if not pages:
        return {1: md_text}
    return {p: "\n".join(ls) for p, ls in sorted(pages.items())}


# ── public API ───────────────────────────────────────────────────────────────

def is_markdown(filename: str | None, content_type: str | None = None) -> bool:
    """True when a filename/MIME type indicates Markdown rather than PDF."""
    name = (filename or "").lower()
    if name.endswith((".md", ".markdown", ".mdown", ".mkd")):
        return True
    return (content_type or "").lower() in ("text/markdown", "text/x-markdown")


def extract_events_from_markdown(md_text: str) -> list[dict]:
    """Parse Markdown tables straight into capital-event dicts.

    De-duplicated within the filing (``merge_with_dedup`` against an empty LLM
    list), since the same allotment is often restated across tables — matching
    the PDF path's ``TableExtractor.extract_events`` behaviour.
    """
    events: list[dict] = []
    for chunk in parse_markdown_tables(md_text):
        events.extend(table_to_events(chunk))
    return merge_with_dedup(events, [])


def extract_events_from_md_file(path: str | Path) -> list[dict]:
    """Read a ``.md`` file and extract its capital events."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return extract_events_from_markdown(text)


def markdown_summary(md_text: str) -> dict:
    """Document summary shaped like ``ocr.process_document`` for the ingest job.

    There is no OCR and no rendering for Markdown, so ``ocr_page_count`` is 0
    and every page is method ``"markdown"``; ``text_by_page`` lets the optional
    ``?llm=true`` pass reuse the same code path as the PDF route.
    """
    pages = split_into_pages(md_text)
    return {
        "total_pages": len(pages),
        "ocr_page_count": 0,
        "text_page_count": len(pages),
        "pages": [
            {"page_num": p, "method": "markdown", "confidence": 1.0, "chars": len(t)}
            for p, t in sorted(pages.items())
        ],
        "text_by_page": pages,
    }
