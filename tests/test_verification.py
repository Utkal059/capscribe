"""Consistency-verification tests.

Covers the three contradiction checks (timeline, capital continuity, bonus
arithmetic) on a clean dataset and on datasets with each kind of planted
error, plus the ratio parser and the report summary.
"""
from __future__ import annotations

from verification import (
    ratio_value,
    verify_consistency,
    verify_report,
)

CONSISTENT = [
    {"event_type": "authorised_capital_change", "date": "2022-01-01",
     "old_capital": 100000000, "new_capital": 600000000, "event_id": "a1"},
    {"event_type": "authorised_capital_change", "date": "2023-01-01",
     "old_capital": 600000000, "new_capital": 1500000000, "event_id": "a2"},
    {"event_type": "bonus_issue", "date": "2021-08-01", "ratio": "5:1",
     "pre_issue_capital": 100000000, "post_issue_capital": 600000000, "event_id": "b1"},
    {"event_type": "allotment", "date": "2017-02-03", "shares": 10000, "event_id": "al1"},
]


def test_clean_dataset_is_consistent():
    res = verify_consistency(CONSISTENT)
    assert res.consistent
    assert res.issues == []
    assert res.confidence == 1.0


def test_capital_continuity_break_flagged():
    broken = [
        {"event_type": "authorised_capital_change", "date": "2022-01-01",
         "old_capital": 100000000, "new_capital": 600000000},
        {"event_type": "authorised_capital_change", "date": "2023-01-01",
         "old_capital": 500000000, "new_capital": 1500000000},  # 600M != 500M
    ]
    res = verify_consistency(broken)
    assert not res.consistent
    assert any(i.check_type == "capital_continuity" for i in res.issues)
    assert all(i.severity in {"warning", "error"} for i in res.issues)


def test_conflicting_same_date_flagged():
    dup = [
        {"event_type": "allotment", "date": "2020-01-01", "shares": 100, "event_id": "x"},
        {"event_type": "allotment", "date": "2020-01-01", "shares": 200, "event_id": "y"},
    ]
    res = verify_consistency(dup)
    assert not res.consistent
    issue = next(i for i in res.issues if i.check_type == "timeline")
    assert set(issue.event_ids) == {"x", "y"}


def test_same_date_same_price_is_multitranche_warning():
    # two allotments, same date, same issue price, different allottees/shares
    # -> a multi-tranche funding round, downgraded from error to warning.
    multi = [
        {"event_type": "allotment", "date": "2023-09-22", "shares": 6358765,
         "issue_price": 129.9, "event_id": "m1"},
        {"event_type": "allotment", "date": "2023-09-22", "shares": 6485940,
         "issue_price": 129.9, "event_id": "m2"},
    ]
    res = verify_consistency(multi)
    issue = next(i for i in res.issues if i.check_type == "timeline")
    assert issue.severity == "warning"
    assert "multi-tranche" in issue.description.lower()


def test_bonus_arithmetic_deviation_flagged():
    bad = [{"event_type": "bonus_issue", "date": "2021-01-01", "ratio": "5:1",
            "pre_issue_capital": 100000000, "post_issue_capital": 900000000}]
    res = verify_consistency(bad)
    assert any(i.check_type == "arithmetic" for i in res.issues)


def test_bonus_arithmetic_within_tolerance_passes():
    ok = [{"event_type": "bonus_issue", "date": "2021-01-01", "ratio": "5:1",
           "pre_issue_capital": 100000000, "post_issue_capital": 600000000}]
    assert verify_consistency(ok).consistent


def test_ratio_value():
    assert ratio_value("5:1") == 5.0
    assert ratio_value("1:4") == 0.25
    assert ratio_value("garbage") is None
    assert ratio_value(None) is None


def test_scope_filter_by_event_type():
    # restricting to bonus_issue ignores the capital-continuity break
    broken = [
        {"event_type": "authorised_capital_change", "date": "2022-01-01",
         "old_capital": 1, "new_capital": 2},
        {"event_type": "authorised_capital_change", "date": "2023-01-01",
         "old_capital": 99, "new_capital": 100},
    ]
    assert verify_consistency(broken, event_type="bonus_issue").consistent


def test_verify_report_summary():
    rep = verify_report(CONSISTENT)
    assert rep.checked == len(CONSISTENT)
    assert rep.consistent
    assert rep.by_check == {}
