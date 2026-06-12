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


def test_ingest_rejects_non_pdf(store):
    r = _client(store).post(
        "/ingest", files={"file": ("notes.txt", b"not a pdf", "text/plain")}
    )
    assert r.status_code == 400


def test_ingest_status_unknown_job(store):
    r = _client(store).get("/ingest/status/does-not-exist")
    assert r.status_code == 404
