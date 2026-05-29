from retrieval import event_to_text


def test_index_counts_all_events(store, sample_events):
    assert store.count() == len(sample_events)


def test_stats_groups_by_type(store):
    stats = store.stats()
    assert stats["total"] == sum(stats["by_type"].values())
    assert stats["by_type"]["allotment"] == 3


def test_search_returns_scored_hits(store):
    # Query with the exact rendered text so the stub embedding matches.
    bonus_text = event_to_text(
        {"event_type": "bonus_issue", "date": "2021-08-01", "ratio": "5:1"}
    )
    hits = store.search(bonus_text, k=1)
    assert hits
    assert hits[0].event["event_type"] == "bonus_issue"
    assert 0.0 <= hits[0].score <= 1.0


def test_event_to_text_is_human_readable():
    text = event_to_text(
        {"event_type": "allotment", "date": "2021-01-01", "allottee_category": "promoters"}
    )
    assert "allotment" in text and "promoters" in text
