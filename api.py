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
    POST /ingest          PDF upload -> OCR fallback + table extraction
    POST /index           rebuild the index from a different extracted JSON

The index is built once on startup from settings.events_path using local
embeddings, so the server boots and serves search/ask with no API spend.
Only POST /ask with mode="llm" calls Claude.
"""
from __future__ import annotations

import logging
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


def _build(events_path: str) -> int:
    store = EventStore()  # default local embeddings, persistent
    n = store.index_events(load_events(events_path))
    hybrid = HybridRetriever(store)
    hybrid.save(BM25_PICKLE)
    state["store"] = store
    state["hybrid"] = hybrid
    state["agent"] = CapScribeAgent(hybrid)
    state["source"] = events_path
    logger.info("indexed %d events from %s (hybrid BM25+vector)", n, events_path)
    return n


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


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)) -> dict:
    """Ingest a PDF: OCR-fallback text extraction → table extraction → summary.

    Runs synchronously and records the result under a job id so the result
    can also be re-fetched from ``GET /ingest/status/{job_id}``. LLM
    extraction is intentionally not invoked here (no API spend); table +
    OCR provenance is returned directly.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "a .pdf file is required")

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"job_id": job_id, "filename": file.filename, "status": "processing"}
    tmp_path = Path(tempfile.gettempdir()) / f"capscribe_{job_id}.pdf"
    try:
        tmp_path.write_bytes(await file.read())
        doc = process_document(tmp_path)
        table_events = TableExtractor().extract_events(tmp_path)
        merged = merge_with_dedup(table_events, [])
        method_counts: dict[str, int] = {}
        for ev in merged:
            m = (ev.get("source_provenance") or {}).get("extraction_method", "table")
            method_counts[m] = method_counts.get(m, 0) + 1
        result = {
            "job_id": job_id,
            "filename": file.filename,
            "status": "done",
            "total_pages": doc["total_pages"],
            "ocr_page_count": doc["ocr_page_count"],
            "text_page_count": doc["text_page_count"],
            "page_methods": doc["pages"],
            "events_extracted": len(merged),
            "events_by_method": method_counts,
            "events": merged,  # the actual extracted events, so the UI can render them
            "ocr_available": ocr_available(),
        }
        JOBS[job_id] = result
        return result
    except Exception as exc:  # surface failures as a job status, not a 500
        JOBS[job_id] = {"job_id": job_id, "filename": file.filename,
                        "status": "error", "error": str(exc)}
        raise HTTPException(500, f"ingestion failed: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str) -> dict:
    """Return the recorded status/result for an ingest job."""
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, f"unknown job: {job_id}")
    return job


@app.get("/")
def home() -> FileResponse:
    return FileResponse("frontend/index.html")