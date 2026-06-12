"""Table-extraction tests.

Exercise the pure logic — value parsing, column detection per table family,
and table-vs-LLM dedup — with synthetic :class:`TableChunk`s, so no PDF or
pdfplumber rendering is needed.
"""
from __future__ import annotations

from table_extractor import (
    TableChunk,
    merge_with_dedup,
    parse_amount,
    parse_date,
    parse_int,
    table_to_events,
)


def _chunk(headers, rows, page=5, section="Capital Build-up"):
    return TableChunk(
        page_num=page,
        bbox=(10.0, 20.0, 100.0, 200.0),
        headers=[h.lower() for h in headers],
        rows=rows,
        raw_text="raw table text",
        source_section=section,
    )


# ── value parsing ──────────────────────────────────────────────────────────────

def test_parse_amount_indian_grouping():
    assert parse_amount("2,50,000") == 250000.0
    assert parse_amount("Rs. 154.20") == 154.2
    assert parse_amount("1,50,00,00,000") == 1500000000.0


def test_parse_amount_empty():
    assert parse_amount("—") is None
    assert parse_amount(None) is None
    assert parse_int("12 shares") == 12


def test_parse_date_formats():
    assert parse_date("2017-02-03") == "2017-02-03"
    assert parse_date("February 3, 2017") == "2017-02-03"
    assert parse_date("") is None
    assert parse_date("not a date") == "not a date"  # falls back to raw


# ── allotment table ────────────────────────────────────────────────────────────

def test_allotment_table_to_events():
    headers = ["date of allotment", "no. of shares", "face value", "issue price", "consideration"]
    rows = [{
        "date of allotment": "2017-02-03", "no. of shares": "10,000",
        "face value": "10", "issue price": "154.20", "consideration": "Cash",
    }]
    evs = table_to_events(_chunk(headers, rows))
    assert len(evs) == 1
    e = evs[0]
    assert e["event_type"] == "allotment"
    assert e["shares"] == 10000
    assert e["date"] == "2017-02-03"
    assert e["issue_price"] == 154.2
    assert e["page_number"] == 5
    assert e["source_provenance"]["extraction_method"] == "table"
    assert e["source_provenance"]["section"] == "Capital Build-up"
    assert e["bbox"] == [10.0, 20.0, 100.0, 200.0]


# ── bonus table ────────────────────────────────────────────────────────────────

def test_bonus_table_to_events():
    headers = ["date", "ratio", "record date"]
    rows = [{"date": "2021-08-01", "ratio": "5:1", "record date": "2021-08-10"}]
    evs = table_to_events(_chunk(headers, rows))
    assert evs and evs[0]["event_type"] == "bonus_issue"
    assert evs[0]["ratio"] == "5:1"


# ── authorised capital table ───────────────────────────────────────────────────

def test_authorised_capital_table_to_events():
    headers = ["date of resolution", "increased from", "increased to", "type of resolution"]
    rows = [{
        "date of resolution": "2023-01-10", "increased from": "60,00,00,000",
        "increased to": "1,50,00,00,000", "type of resolution": "Special Resolution",
    }]
    evs = table_to_events(_chunk(headers, rows))
    assert len(evs) == 1
    e = evs[0]
    assert e["event_type"] == "authorised_capital_change"
    assert e["old_capital"] == 600000000
    assert e["new_capital"] == 1500000000
    assert e["resolution_type"] == "special resolution"


# ── non-matching tables ────────────────────────────────────────────────────────

def test_unknown_table_yields_nothing():
    assert table_to_events(_chunk(["foo", "bar"], [{"foo": "1", "bar": "2"}])) == []


def test_empty_table_yields_nothing():
    assert table_to_events(_chunk([], [])) == []


# ── dedup ──────────────────────────────────────────────────────────────────────

def test_merge_prefers_table_version():
    llm = [{"event_type": "allotment", "date": "2017-02-03", "shares": 10000,
            "source_provenance": {"extraction_method": "llm"}}]
    table = [{"event_type": "allotment", "date": "2017-02-03", "shares": 10000,
              "source_provenance": {"extraction_method": "table"}}]
    merged = merge_with_dedup(table, llm)
    assert len(merged) == 1
    assert merged[0]["source_provenance"]["extraction_method"] == "table"


def test_merge_keeps_distinct_events():
    llm = [{"event_type": "allotment", "date": "2019-06-21", "shares": 250000}]
    table = [{"event_type": "allotment", "date": "2017-02-03", "shares": 10000}]
    assert len(merge_with_dedup(table, llm)) == 2
