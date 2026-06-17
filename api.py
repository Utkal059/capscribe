"""FastAPI service for CapScribe.

Ships the extraction output as a queryable internal tool — the "APIs and
internal tools using FastAPI" line item in the JD. Endpoints:

    GET  /health          liveness
    GET  /stats           event counts by type
    GET  /events          list events (optional ?event_type= & ?limit=)
    POST /search          semantic search  {query, k}
    POST /ask             agentic RAG      {question, mode}
    POST /verify          full-corpus contradiction report
    POST /report          source-backed capital-history brief {mode, title}
    POST /ingest          PDF upload -> background OCR + table extraction (?llm=true adds Claude)
    GET  /ingest/status/{job_id}   poll an ingest job
    POST /ingest/{job_id}/index    promote an ingested filing to the live corpus
    POST /index           rebuild the index from a different extracted JSON

The index is built once on startup from settings.events_path using local
embeddings, so the server boots and serves search/ask with no API spend.
Only POST /ask with mode="llm" calls Claude.
"""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from pathlib import Path

from agent import CapScribeAgent
from config import settings
from ocr import ocr_available, process_document
from report import generate_report
from retrieval import EventStore, HybridRetriever, load_events
from table_extractor import TableExtractor, merge_with_dedup
from verification import events_from_store, verify_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("capscribe.api")

BM25_PICKLE = Path(settings.chroma_path) / "bm25_index.pkl"


@asynccontextmanager
async def lifespan(_: "FastAPI"):
    _build(settings.events_path)
    yield


app = FastAPI(title="CapScribe API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state: dict = {"store": None, "agent": None, "source": None}

# In-memory ingest job registry (no external queue needed for the demo).
JOBS: dict[str, dict] = {}

# Demo guard rails: a large filing (100+ pages) OOM-crashes a small 512 MB
# instance mid-extraction. We reject oversized uploads up front (the page count
# is read cheaply with pypdf, before the heavy extraction) so the user gets a
# clean message instead of a 502. Raise these via env on a bigger plan.
MAX_INGEST_PAGES = int(os.getenv("MAX_INGEST_PAGES", "40"))
MAX_INGEST_MB = float(os.getenv("MAX_INGEST_MB", "6"))


class SearchRequest(BaseModel):
    query: str
    k: int | None = None
    # fusion weight: 0 = pure BM25, 1 = pure vector, None = auto heuristic
    alpha: float | None = None


class AskRequest(BaseModel):
    question: str
    mode: str = "extractive"  # "extractive" (free) | "llm" (uses Claude)


class IndexRequest(BaseModel):
    events_path: str


class ReportRequest(BaseModel):
    mode: str = "extractive"  # "extractive" (free) | "llm" (Claude summary)
    title: str | None = None


def _index_events(events: list[dict], source: str) -> int:
    """Rebuild the live corpus (vector + BM25 + agent) from event dicts.

    Shared by startup (from the fixture JSON) and by ``/ingest/{id}/index``
    (from a freshly uploaded filing), so an ingested PDF can become the
    corpus that search / ask / verify / report all answer against.
    """
    store = EventStore()  # default local embeddings, persistent
    n = store.index_events(events)
    hybrid = HybridRetriever(store)
    hybrid.save(BM25_PICKLE)
    state["store"] = store
    state["hybrid"] = hybrid
    state["agent"] = CapScribeAgent(hybrid)
    state["source"] = source
    logger.info("indexed %d events from %s (hybrid BM25+vector)", n, source)
    return n


def _build(events_path: str) -> int:
    return _index_events(load_events(events_path), events_path)


def _retriever():
    """Hybrid when built; falls back to the plain vector store."""
    return state.get("hybrid") or state["store"]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "source": state["source"],
        "indexed": state["store"].count(),
        "retrieval": "hybrid" if state.get("hybrid") else "vector",
        "ocr_available": ocr_available(),
        # Whether OCR actually ran on any page of the most recent ingest
        # (the JD names "scans" explicitly — this makes the fallback visible).
        "ocr_last_ingest": state.get("ocr_last_ingest"),
    }


@app.get("/stats")
def stats() -> dict:
    return state["store"].stats()


@app.get("/events")
def events(event_type: str | None = None, limit: int = 100) -> dict:
    res = state["store"]._collection.get()
    metas = res.get("metadatas", []) or []
    if event_type:
        metas = [m for m in metas if m.get("event_type") == event_type]
    return {"count": len(metas[:limit]), "events": metas[:limit]}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(400, "query must not be empty")
    retriever = _retriever()
    if isinstance(retriever, HybridRetriever):
        hits = retriever.search(req.query, k=req.k, alpha=req.alpha)
        strategy = retriever.last_strategy
    else:
        hits = retriever.search(req.query, k=req.k)
        strategy = "vector"
    return {
        "query": req.query,
        "hits": [h.model_dump() for h in hits],
        "retrieval_strategy": strategy,
    }


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if req.mode not in ("extractive", "llm"):
        raise HTTPException(400, "mode must be 'extractive' or 'llm'")
    return state["agent"].ask(req.question, mode=req.mode).model_dump()


@app.post("/index")
def index(req: IndexRequest) -> dict:
    try:
        n = _build(req.events_path)
    except FileNotFoundError:
        raise HTTPException(404, f"file not found: {req.events_path}")
    return {"indexed": n, "source": req.events_path}


@app.post("/verify")
def verify() -> dict:
    """Run full-corpus consistency verification (timeline / continuity / arithmetic)."""
    events = events_from_store(_retriever())
    return verify_report(events).model_dump()


@app.post("/report")
def report(req: ReportRequest) -> dict:
    """Generate a source-backed capital-history brief over the indexed filing.

    ``mode="extractive"`` (default) is free and deterministic; ``mode="llm"``
    adds a Claude-written executive summary over the same computed facts.
    Returns structured fields plus a rendered Markdown brief.
    """
    if req.mode not in ("extractive", "llm"):
        raise HTTPException(400, "mode must be 'extractive' or 'llm'")
    events = events_from_store(_retriever())
    return generate_report(events, mode=req.mode, title=req.title).model_dump()


def _llm_events(text_by_page: dict[int, str], max_pages: int = 30) -> list[dict]:
    """Optional LLM extraction over capital-relevant pages (opt-in; costs API).

    Bounded to the first ``max_pages`` pages that carry capital-event signals
    so a 400-page filing can't trigger hundreds of paid calls. Returns []
    (with a warning) when no API key is configured, so the free table path
    always still works.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.warning("LLM ingest requested but ANTHROPIC_API_KEY is unset; skipping")
        return []
    from extractor import attach_provenance, call_claude  # lazy: keeps free path import-light
    from table_extractor import CAPITAL_SIGNALS

    relevant = sorted(
        p for p, t in text_by_page.items()
        if any(s in (t or "").lower() for s in CAPITAL_SIGNALS)
    )[:max_pages]
    events: list[dict] = []
    for p in relevant:
        try:
            events.extend(call_claude(f"[PAGE {p}]\n{text_by_page[p]}"))
        except Exception as exc:  # one bad page shouldn't sink the whole pass
            logger.warning("LLM extraction failed on page %s: %s", p, exc)
    events = attach_provenance(events, text_by_page)
    for ev in events:  # tag origin so /ingest method counts stay honest
        prov = ev.setdefault("source_provenance", {})
        prov["extraction_method"] = "llm"
    logger.info("LLM pass: %d events over %d capital pages", len(events), len(relevant))
    return events


def _process_ingest(job_id: str, tmp_path: str, filename: str, use_llm: bool = False) -> None:
    """Run the heavy PDF work (OCR scan + table extraction) for one job.

    Synchronous and CPU-bound, so it runs in a worker thread (see ``/ingest``)
    rather than on the event loop. Results — including the extracted ``events``
    array the UI renders — are written back into ``JOBS[job_id]``. When
    ``use_llm`` is set, a bounded Claude pass is merged in (table wins on a
    fuzzy match) to also catch prose-narrated events.
    """
    path = Path(tmp_path)
    try:
        doc = process_document(path)
        # Surface the OCR fallback: log it and record it for /health so a
        # reviewer can see scans were handled (or that none were needed).
        ocr_pages = doc["ocr_page_count"]
        logger.info("ingest %s: OCR ran on %d/%d page(s) (ocr_available=%s)",
                    filename, ocr_pages, doc["total_pages"], ocr_available())
        state["ocr_last_ingest"] = {
            "file": filename, "ocr_pages": ocr_pages,
            "total_pages": doc["total_pages"], "ran": ocr_pages > 0,
        }
        table_events = TableExtractor().extract_events(path)
        llm_events = _llm_events(doc["text_by_page"]) if use_llm else []
        merged = merge_with_dedup(table_events, llm_events)
        method_counts: dict[str, int] = {}
        for ev in merged:
            m = (ev.get("source_provenance") or {}).get("extraction_method", "table")
            method_counts[m] = method_counts.get(m, 0) + 1
        JOBS[job_id] = {
            "job_id": job_id,
            "filename": filename,
            "status": "done",
            "total_pages": doc["total_pages"],
            "ocr_page_count": doc["ocr_page_count"],
            "text_page_count": doc["text_page_count"],
            "page_methods": doc["pages"],
            "events_extracted": len(merged),
            "events_by_method": method_counts,
            "events": merged,  # the actual events — the UI reads data.events
            "ocr_available": ocr_available(),
        }
        logger.info("ingest job %s done: %d events from %s", job_id, len(merged), filename)
    except Exception as exc:  # record the failure on the job, never crash the worker
        logger.exception("ingest job %s failed", job_id)
        JOBS[job_id] = {"job_id": job_id, "filename": filename,
                        "status": "error", "error": str(exc)}
    finally:
        path.unlink(missing_ok=True)


@app.post("/ingest")
async def ingest(file: UploadFile = File(...), llm: bool = False) -> dict:
    """Accept a PDF, start extraction in the background, return a job id.

    A large filing takes a couple of minutes to parse, so the work runs in a
    worker thread and the client polls ``GET /ingest/status/{job_id}`` for the
    result. Returning immediately keeps the event loop (and every other
    request) responsive and avoids upstream gateway timeouts.

    ``?llm=true`` additionally runs a bounded Claude extraction pass (uses API
    credits); the default is the free, deterministic table path only.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "a .pdf file is required")

    data = await file.read()
    size_mb = len(data) / 1_000_000
    if size_mb > MAX_INGEST_MB:
        logger.warning("ingest guard: rejected %s (%.1f MB > %.0f MB limit)",
                       file.filename, size_mb, MAX_INGEST_MB)
        raise HTTPException(413,
            f"This filing is {size_mb:.0f} MB — the live demo instance is memory-limited "
            f"to {MAX_INGEST_MB:.0f} MB. Try the capital-structure section, or explore the "
            f"pre-loaded Ola DRHP via Search / Ask / Report.")

    job_id = uuid.uuid4().hex[:12]
    tmp_path = Path(tempfile.gettempdir()) / f"capscribe_{job_id}.pdf"
    tmp_path.write_bytes(data)

    # Page-count guard: counting pages with pypdf is cheap and low-memory, so we
    # can reject a too-large filing here, before the heavy extraction that would
    # OOM-crash the instance.
    try:
        from pypdf import PdfReader
        n_pages = len(PdfReader(str(tmp_path)).pages)
    except Exception:
        n_pages = 0
    if n_pages > MAX_INGEST_PAGES:
        logger.warning("ingest guard: rejected %s (%d pages > %d limit)",
                       file.filename, n_pages, MAX_INGEST_PAGES)
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(413,
            f"This filing has {n_pages} pages — the live demo instance handles up to "
            f"{MAX_INGEST_PAGES}. Upload just the capital-structure section, or explore the "
            f"pre-loaded Ola DRHP (Search / Ask / Verify / Report all work on it).")

    JOBS[job_id] = {"job_id": job_id, "filename": file.filename, "status": "processing"}

    # Fire-and-forget on the default thread-pool executor; the worker updates
    # JOBS when finished. We don't await it, so the response returns at once.
    asyncio.get_running_loop().run_in_executor(
        None, _process_ingest, job_id, str(tmp_path), file.filename, llm
    )
    return {"job_id": job_id, "filename": file.filename, "status": "processing"}


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str) -> dict:
    """Return the recorded status/result for an ingest job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    return job


@app.post("/ingest/{job_id}/index")
def ingest_index(job_id: str) -> dict:
    """Promote a finished ingest job's events to the live corpus.

    Re-indexes search / ask / verify / report onto the uploaded filing so the
    whole product answers about *that* document instead of the seed fixtures.
    """
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    if job.get("status") != "done":
        raise HTTPException(409, f"job not finished (status: {job.get('status')})")
    events = job.get("events") or []
    if not events:
        raise HTTPException(400, "this job extracted no events to index")
    source = f"ingest:{job.get('filename') or job_id}"
    n = _index_events(events, source)
    return {"indexed": n, "source": source, "job_id": job_id, "stats": state["store"].stats()}


@app.get("/")
def home() -> FileResponse:
    return FileResponse("frontend/index.html")
