"""API tests.

The app's real startup builds an index with the local embedding model
(a one-time download on a dev machine). To keep tests offline we clear the
startup hook and inject the in-memory stub store from conftest instead.
"""
from fastapi.testclient import TestClient

import api
from agent import CapScribeAgent


def _client(store):
    # TestClient only runs the lifespan when used as a context manager, so
    # plain instantiation skips the real (downloading) index build. We inject
    # the in-memory stub store directly instead.
    api.state["store"] = store
    api.state["agent"] = CapScribeAgent(store)
    api.state["source"] = "test"
    return TestClient(api.app)


def test_health(store):
    c = _client(store)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["indexed"] == store.count()


def test_stats(store):
    r = _client(store).get("/stats")
    assert r.status_code == 200
    assert r.json()["by_type"]["allotment"] == 3


def test_events_filter(store):
    r = _client(store).get("/events", params={"event_type": "bonus_issue"})
    assert r.status_code == 200
    assert all(e["event_type"] == "bonus_issue" for e in r.json()["events"])


def test_search_empty_query_rejected(store):
    r = _client(store).post("/search", json={"query": "   "})
    assert r.status_code == 400


def test_ask_extractive(store):
    r = _client(store).post(
        "/ask", json={"question": "bonus issue", "mode": "extractive"}
    )
    assert r.status_code == 200
    assert r.json()["mode"] == "extractive"


def test_ask_bad_mode(store):
    r = _client(store).post("/ask", json={"question": "x", "mode": "nope"})
    assert r.status_code == 400


def test_verify_endpoint(store):
    r = _client(store).post("/verify")
    assert r.status_code == 200
    body = r.json()
    assert body["checked"] == store.count()
    assert isinstance(body["consistent"], bool)
    assert "by_check" in body


def test_report_endpoint(store):
    r = _client(store).post("/report", json={"mode": "extractive"})
    assert r.status_code == 200
    body = r.json()
    assert body["event_count"] == store.count()
    assert body["mode"] == "extractive"
    assert body["markdown"].startswith("# Capital History Report")
    assert "verification" in body
    assert isinstance(body["citations"], list)


def test_report_rejects_bad_mode(store):
    r = _client(store).post("/report", json={"mode": "bogus"})
    assert r.status_code == 400


def test_report_custom_title(store):
    r = _client(store).post("/report", json={"title": "Sample Filing — Brief"})
    assert r.status_code == 200
    assert r.json()["title"] == "Sample Filing — Brief"


def test_ingest_rejects_non_pdf(store):
    r = _client(store).post(
        "/ingest", files={"file": ("notes.txt", b"not a pdf", "text/plain")}
    )
    assert r.status_code == 400


def test_ingest_status_unknown_job(store):
    r = _client(store).get("/ingest/status/does-not-exist")
    assert r.status_code == 404


def test_ingest_rejects_oversized_file(store, monkeypatch):
    # over MAX_INGEST_MB -> rejected up front, never reaches extraction. Pin a
    # small limit so the test stays fast and independent of the prod default.
    monkeypatch.setattr(api, "MAX_INGEST_MB", 1.0)
    big = b"%PDF-1.4\n" + b"0" * (2 * 1000 * 1000)  # ~2 MB > 1 MB pinned limit
    r = _client(store).post("/ingest", files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413
    assert "MB" in r.json()["detail"]


def test_ingest_rejects_too_many_pages(store):
    import io
    from pypdf import PdfWriter
    w = PdfWriter()
    for _ in range(api.MAX_INGEST_PAGES + 1):
        w.add_blank_page(width=72, height=72)
    buf = io.BytesIO(); w.write(buf)
    r = _client(store).post("/ingest", files={"file": ("many.pdf", buf.getvalue(), "application/pdf")})
    assert r.status_code == 413
    assert "pages" in r.json()["detail"]


def test_ingest_index_unknown_job(store):
    r = _client(store).post("/ingest/does-not-exist/index")
    assert r.status_code == 404


def test_ingest_index_unfinished_job(store):
    c = _client(store)
    api.JOBS["pending-job"] = {"job_id": "pending-job", "status": "processing"}
    try:
        r = c.post("/ingest/pending-job/index")
        assert r.status_code == 409
    finally:
        api.JOBS.pop("pending-job", None)
