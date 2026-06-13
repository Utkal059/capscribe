"""Capital-history report tests.

The report skeleton is computed deterministically (no API, no PDF, no store),
so these run fully offline. They cover the rollup metrics, chronological
timeline, page-only citation rule, embedded verification, Markdown rendering,
and the empty-input edge case.
"""
from __future__ import annotations

from report import generate_report

EVENTS = [
    {"event_type": "allotment", "date": "2017-02-03", "shares": 10000,
     "issue_price": 10, "allottee_category": "promoters",
     "page_number": 72, "source_snippet": "allotted 10,000 Equity Shares ...",
     "event_id": "al1"},
    {"event_type": "allotment", "date": "2019-06-21", "shares": 50000,
     "issue_price": 154.20, "allottee_category": "investors",
     "page_number": 73, "source_snippet": "preferential allotment at Rs. 154.20 ...",
     "event_id": "al2"},
    {"event_type": "bonus_issue", "date": "2021-08-01", "ratio": "5:1",
     "pre_issue_capital": 100000000, "post_issue_capital": 600000000,
     "page_number": 75, "source_snippet": "bonus issue in the ratio 5:1 ...",
     "event_id": "b1"},
    {"event_type": "authorised_capital_change", "date": "2023-01-10",
     "old_capital": 600000000, "new_capital": 1500000000,
     "resolution_type": "special_resolution",
     "page_number": 78, "source_snippet": "authorised share capital increased ...",
     "event_id": "acc1"},
    {"event_type": "dividend_declaration", "date": "2022-07-15",
     "amount_per_share": 2.5, "total_outflow": 150000000,
     "page_number": 92, "source_snippet": "final dividend of Rs. 2.50 ...",
     "event_id": "d1"},
]


def test_report_basic_shape():
    r = generate_report(EVENTS)
    assert r.event_count == 5
    assert r.mode == "extractive"
    assert r.by_type["allotment"] == 2
    assert r.date_range == ["2017-02-03", "2023-01-10"]


def test_metrics_rollups():
    m = generate_report(EVENTS).metrics
    assert m["total_shares_allotted"] == 60000      # 10000 + 50000
    assert m["allotment_events"] == 2
    assert m["bonus_issues"] == 1
    assert m["latest_authorised_capital"] == 1500000000
    assert m["total_dividend_outflow"] == 150000000


def test_timeline_is_chronological():
    timeline = generate_report(EVENTS).timeline
    dates = [t.date for t in timeline]
    assert dates == sorted(dates)
    assert timeline[0].date == "2017-02-03"


def test_citations_only_when_page_known():
    events = EVENTS + [{"event_type": "allotment", "date": "2024-01-01",
                        "shares": 1, "event_id": "nopage"}]  # no page/snippet
    r = generate_report(events)
    # every cited event has a page; the page-less one is excluded
    assert all(c.page_number for c in r.citations)
    assert len(r.citations) == 5
    assert "nopage" not in {c.event_id for c in r.citations}


def test_verification_embedded():
    r = generate_report(EVENTS)
    assert r.verification["consistent"] is True
    assert r.verification["checked"] == 5


def test_inconsistency_surfaces_in_report():
    broken = EVENTS + [
        {"event_type": "authorised_capital_change", "date": "2024-01-01",
         "old_capital": 999, "new_capital": 2000000000, "event_id": "acc2"},
    ]  # 1,500,000,000 (2023) != 999 (2024 old) -> continuity break
    r = generate_report(broken)
    assert r.verification["consistent"] is False
    assert "capital_continuity" in r.verification["by_check"]


def test_markdown_rendered_with_citations():
    md = generate_report(EVENTS).markdown
    assert md.startswith("# Capital History Report")
    assert "## Capital timeline" in md
    assert "## Verification" in md
    assert "(p. 72)" in md                 # inline page citation present
    assert "Rs. 150.00 Cr" in md           # latest authorised capital formatted


def test_custom_title():
    r = generate_report(EVENTS, title="Ola Electric — Capital Brief")
    assert r.title == "Ola Electric — Capital Brief"
    assert r.markdown.startswith("# Ola Electric — Capital Brief")


def test_empty_events_is_safe():
    r = generate_report([])
    assert r.event_count == 0
    assert r.date_range == [None, None]
    assert r.timeline == []
    assert r.verification["consistent"] is True
    assert r.markdown.startswith("# Capital History Report")
