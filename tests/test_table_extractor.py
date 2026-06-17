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


# ── number parsing robustness ───────────────────────────────────────────────────

def test_parse_amount_currency_and_plain():
    assert parse_amount("₹1,50,000") == 150000.0
    assert parse_amount("INR 154.20") == 154.2
    assert parse_amount("250000") == 250000.0
    assert parse_amount("Nil") is None
    assert parse_amount("N/A") is None


def test_parse_amount_magnitude_words():
    assert parse_amount("Rs. 150 crore") == 1_500_000_000.0
    assert parse_amount("10 lakh") == 1_000_000.0
    assert parse_amount("2.5 crore") == 25_000_000.0
    assert parse_amount("5 mn") == 5_000_000.0


def test_parse_amount_full_figure_with_spelled_word_not_multiplied():
    # full digits + a spelled-out parenthetical must NOT be multiplied
    assert parse_amount("1,50,00,00,000 (Rupees One Hundred Fifty Crore)") == 1_500_000_000.0


def test_parse_date_more_formats():
    assert parse_date("31-01-2019") == "2019-01-31"
    assert parse_date("31.01.2019") == "2019-01-31"
    assert parse_date("31 Jan 2019") == "2019-01-31"
    assert parse_date("31-Jan-2019") == "2019-01-31"
    assert parse_date("January 31 2019") == "2019-01-31"


def test_parse_date_strips_footnote_markers():
    assert parse_date("December 19, 2023*") == "2023-12-19"
    assert parse_date("2019-01-31#") == "2019-01-31"
    assert parse_date("February 3, 2017 (1)") == "2017-02-03"


# ── header-variant tolerance ────────────────────────────────────────────────────

def test_allotment_header_spelling_variants():
    headers = ["Allotment Date", "Number of shares allotted", "Face Value",
               "Issue Price per Equity Share"]
    rows = [{"allotment date": "2021-07-29", "number of shares allotted": "21",
             "face value": "10", "issue price per equity share": "10"}]
    evs = table_to_events(_chunk(headers, rows))
    assert len(evs) == 1 and evs[0]["event_type"] == "allotment"
    assert evs[0]["shares"] == 21
    assert evs[0]["date"] == "2021-07-29"


# ── bonus / rights precision guard ──────────────────────────────────────────────

def test_financial_ratio_table_is_not_a_bonus():
    # a debt-equity / current-ratio table has a "ratio" column but no record or
    # allotment date -> must yield nothing (the precision guard).
    headers = ["Particulars", "Ratio"]
    rows = [{"particulars": "Debt-Equity Ratio", "ratio": "2:1"},
            {"particulars": "Current Ratio", "ratio": "1:1"}]
    assert table_to_events(_chunk(headers, rows, section="Financial Ratios")) == []


def test_bonus_still_detected_with_record_date():
    headers = ["record date", "ratio"]
    rows = [{"record date": "2021-08-10", "ratio": "5:1"}]
    evs = table_to_events(_chunk(headers, rows, section="Bonus Issue"))
    assert len(evs) == 1 and evs[0]["event_type"] == "bonus_issue"
    assert evs[0]["ratio"] == "5:1"


def test_rights_issue_detected_from_section():
    headers = ["record date", "ratio", "issue price"]
    rows = [{"record date": "2022-03-15", "ratio": "1:4", "issue price": "220.50"}]
    evs = table_to_events(_chunk(headers, rows, section="Rights Issue"))
    assert len(evs) == 1 and evs[0]["event_type"] == "rights_issue"
    assert evs[0]["ratio"] == "1:4"
    assert evs[0]["price"] == 220.5


# ── multi-line / wrapped headers (real DRHP variation) ───────────────────────────

def test_multiline_header_merge():
    from table_extractor import _merge_multiline_header
    # header "Date of Allotment" wrapped across two physical rows, as pdfplumber
    # emits it on many real filings, then a real data row.
    data = [
        ["Date of", "Number of", "Face Value", "Nature of"],
        ["Allotment of Equity Shares", "Equity Shares allotted", "per share", "consideration"],
        ["2021-07-29", "21", "10", "Cash"],
    ]
    headers, body = _merge_multiline_header(data)
    assert "date of allotment" in headers[0]
    assert "number of equity shares" in headers[1]
    assert len(body) == 1 and body[0][0] == "2021-07-29"


def test_wrapped_header_share_count_not_taken_from_date_column():
    # the date header itself contains "shares"; the share count must come from
    # the real count column, never a parse of the date column.
    headers = ["date of allotment of equity shares",
               "number of equity shares allotted", "issue price per equity share"]
    rows = [{"date of allotment of equity shares": "2024-04-03",
             "number of equity shares allotted": "1,200",
             "issue price per equity share": "10"}]
    evs = table_to_events(_chunk(headers, rows))
    assert len(evs) == 1 and evs[0]["event_type"] == "allotment"
    assert evs[0]["shares"] == 1200
    assert evs[0]["date"] == "2024-04-03"


def test_cumulative_column_not_used_as_share_count():
    headers = ["date of allotment", "number of equity shares allotted",
               "cumulative number of equity shares"]
    rows = [{"date of allotment": "2024-04-03",
             "number of equity shares allotted": "1,200",
             "cumulative number of equity shares": "9,99,999"}]
    evs = table_to_events(_chunk(headers, rows))
    assert evs[0]["shares"] == 1200  # the allotted count, not the cumulative


# ── equity-only precision guard ──────────────────────────────────────────────────

def test_preference_share_table_is_not_an_equity_allotment():
    # a "number of preference shares" table is a different instrument -> excluded
    headers = ["date of allotment", "number of preference shares",
               "issue price per preference share"]
    rows = [{"date of allotment": "2022-01-22",
             "number of preference shares": "8,475,877",
             "issue price per preference share": "100"}]
    assert table_to_events(_chunk(headers, rows)) == []


def test_mixed_equity_ccps_column_still_extracts():
    # "equity shares / CCPS transacted" mentions equity -> kept
    headers = ["date of allotment", "number of equity shares / ccps transacted",
               "nature of transaction"]
    rows = [{"date of allotment": "2023-07-14",
             "number of equity shares / ccps transacted": "3,179,382",
             "nature of transaction": "Allotment"}]
    evs = table_to_events(_chunk(headers, rows))
    assert len(evs) == 1 and evs[0]["shares"] == 3179382


def test_allotment_rejects_unparseable_date():
    # a wrapped/split date with no year must not produce an event
    headers = ["date of allotment", "number of equity shares allotted"]
    rows = [{"date of allotment": "December 23", "number of equity shares allotted": "1,000"}]
    assert table_to_events(_chunk(headers, rows)) == []
