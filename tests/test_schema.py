from schema import CapitalEvent, ExtractionResult


def test_event_parses_and_keeps_extra_fields():
    ev = CapitalEvent(event_type="allotment", date="2021-01-01", shares=100, weird="x")
    dumped = ev.model_dump()
    assert dumped["event_type"] == "allotment"
    assert dumped["weird"] == "x"  # extra="allow" preserves unmodelled fields


def test_dedup_key_distinguishes_events():
    a = CapitalEvent(event_type="allotment", date="2021-01-01", shares=100)
    b = CapitalEvent(event_type="allotment", date="2021-01-01", shares=200)
    c = CapitalEvent(event_type="allotment", date="2021-01-01", shares=100)
    assert a.dedup_key() != b.dedup_key()
    assert a.dedup_key() == c.dedup_key()


def test_extraction_result_loads(sample_events):
    result = ExtractionResult(source_file="x.pdf", capital_events=sample_events)
    assert len(result.capital_events) == len(sample_events)
    assert result.capital_events[0].event_type in {
        "allotment",
        "bonus_issue",
        "rights_issue",
        "authorised_capital_change",
    }
