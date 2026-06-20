"""Markdown table-extraction tests.

Markdown DRHPs are parsed straight into the same events the PDF path produces,
reusing ``table_to_events``. These exercise the Markdown-specific parsing
(table detection, header/section/page tracking, alignment rows, escaped pipes)
and confirm the shared precision guards still hold on Markdown input — all
without any PDF, pdfplumber, or network.
"""
from __future__ import annotations

import time

from markdown_extractor import (
    extract_events_from_markdown,
    extract_events_from_md_file,
    is_markdown,
    markdown_summary,
    parse_markdown_tables,
    split_into_pages,
)


# ── file-type detection ─────────────────────────────────────────────────────────

def test_is_markdown_by_extension_and_mime():
    assert is_markdown("ola_drhp.md")
    assert is_markdown("FILING.MARKDOWN")
    assert is_markdown("filing.pdf", "text/markdown")  # MIME wins over extension
    assert not is_markdown("ola_drhp.pdf")
    assert not is_markdown("notes.txt")
    assert not is_markdown(None)


# ── table parsing ────────────────────────────────────────────────────────────────

ALLOTMENT_MD = """\
## History of Equity Share Capital

| Date of Allotment | No. of Equity Shares | Face Value | Issue Price |
| --- | ---: | :---: | --- |
| December 19, 2023 | 1,00,000 | 10 | 154.20 |
| January 31, 2019  | 2,50,000 | 10 | 200 |
"""


def test_allotment_table_maps_to_events():
    events = extract_events_from_markdown(ALLOTMENT_MD)
    assert len(events) == 2
    ev = next(e for e in events if e["date"] == "2023-12-19")
    assert ev["event_type"] == "allotment"
    assert ev["shares"] == 100000
    assert ev["face_value"] == 10.0
    assert ev["issue_price"] == 154.2
    # provenance: section heading captured, method is structural "table"
    assert ev["source_provenance"]["section"] == "History of Equity Share Capital"
    assert ev["source_provenance"]["extraction_method"] == "table"


def test_section_and_page_marker_drive_provenance():
    md = (
        "[PAGE 42]\n"
        "## Bonus Issue\n\n"
        "| Ratio | Record Date |\n"
        "| --- | --- |\n"
        "| 2:1 | 2022-03-15 |\n"
    )
    events = extract_events_from_markdown(md)
    assert len(events) == 1
    assert events[0]["event_type"] == "bonus_issue"
    assert events[0]["ratio"] == "2:1"
    assert events[0]["page_number"] == 42


def test_rights_split_from_bonus_on_section_signal():
    md = (
        "## Rights Issue\n\n"
        "| Ratio | Record Date | Issue Price |\n"
        "| --- | --- | --- |\n"
        "| 1:5 | 2021-06-01 | 120 |\n"
    )
    events = extract_events_from_markdown(md)
    assert events[0]["event_type"] == "rights_issue"
    assert events[0]["price"] == 120.0


def test_escaped_pipe_in_cell_is_preserved():
    md = (
        "| Date of Allotment | No. of Shares | Consideration |\n"
        "| --- | --- | --- |\n"
        "| 2020-01-01 | 1,000 | cash \\| premium |\n"
    )
    events = extract_events_from_markdown(md)
    assert events[0]["consideration"] == "cash | premium"


def test_multiple_tables_in_one_document():
    md = (
        "## History of Equity Share Capital\n\n"
        "| Date of Allotment | No. of Equity Shares |\n"
        "| --- | --- |\n"
        "| 2023-01-10 | 1,000 |\n\n"
        "## Increase in Authorised Capital\n\n"
        "| Date of Resolution | Increased From | Increased To |\n"
        "| --- | --- | --- |\n"
        "| 2022-05-01 | 5,00,00,000 | 10,00,00,000 |\n"
    )
    chunks = parse_markdown_tables(md)
    assert len(chunks) == 2
    events = extract_events_from_markdown(md)
    types = sorted(e["event_type"] for e in events)
    assert types == ["allotment", "authorised_capital_change"]


def test_alignment_row_not_treated_as_data():
    # The |:---|:---:| delimiter must be consumed as the separator, never
    # surface as a phantom data row.
    md = (
        "| Date of Allotment | No. of Shares |\n"
        "|:---|:---:|\n"
        "| 2023-01-10 | 5,000 |\n"
    )
    chunk = parse_markdown_tables(md)[0]
    assert len(chunk.rows) == 1  # only the real data row, not the delimiter
    events = extract_events_from_markdown(md)
    assert len(events) == 1
    assert events[0]["shares"] == 5000


# ── precision: non-capital tables stay out ──────────────────────────────────────

def test_financial_ratio_table_is_not_an_event():
    # A debt-equity ratio table has a ratio column but no record/allotment date,
    # so the bonus/rights guard must reject it — same as the PDF path.
    md = (
        "## Key Financial Ratios\n\n"
        "| Metric | Ratio |\n"
        "| --- | --- |\n"
        "| Debt-Equity | 2:1 |\n"
    )
    assert extract_events_from_markdown(md) == []


def test_prose_without_tables_yields_nothing():
    md = "# Risk Factors\n\nThe company may issue equity shares in the future.\n"
    assert parse_markdown_tables(md) == []
    assert extract_events_from_markdown(md) == []


def test_empty_markdown_returns_empty():
    assert extract_events_from_markdown("") == []
    assert parse_markdown_tables("") == []


def test_extract_events_from_md_file(tmp_path):
    p = tmp_path / "filing.md"
    p.write_text(ALLOTMENT_MD, encoding="utf-8")
    events = extract_events_from_md_file(str(p))
    assert len(events) == 2
    assert {e["event_type"] for e in events} == {"allotment"}


# ── document summary / pagination ────────────────────────────────────────────────

def test_split_into_pages_defaults_to_one_page():
    assert list(split_into_pages("no markers here").keys()) == [1]


def test_markdown_summary_shape():
    md = "[PAGE 1]\nintro\n[PAGE 2]\n| A | B |\n| - | - |\n| 1 | 2 |\n"
    doc = markdown_summary(md)
    assert doc["total_pages"] == 2
    assert doc["ocr_page_count"] == 0
    assert doc["text_page_count"] == 2
    assert set(doc["text_by_page"].keys()) == {1, 2}
    assert all(p["method"] == "markdown" for p in doc["pages"])


# ── API: /ingest accepts markdown ────────────────────────────────────────────────

def test_ingest_accepts_markdown_upload(store):
    """POST /ingest with a .md filing returns 200 and the background job
    extracts at least one event; a .txt upload is rejected with 400."""
    from fastapi.testclient import TestClient

    import api
    from agent import CapScribeAgent

    api.state["store"] = store
    api.state["agent"] = CapScribeAgent(store)
    api.state["source"] = "test"
    client = TestClient(api.app)

    r = client.post(
        "/ingest",
        files={"file": ("filing.md", ALLOTMENT_MD, "text/markdown")},
    )
    assert r.status_code == 200
    job_id = r.json()["job_id"]

    # The work runs in a background thread; poll the status endpoint for it.
    job = {}
    for _ in range(100):
        job = client.get(f"/ingest/status/{job_id}").json()
        if job["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert job["status"] == "done"
    assert job["events_extracted"] >= 1
    assert job["ocr_page_count"] == 0  # markdown never touches OCR

    # A non-PDF / non-Markdown upload is rejected up front.
    bad = client.post("/ingest", files={"file": ("notes.txt", "hello", "text/plain")})
    assert bad.status_code == 400
