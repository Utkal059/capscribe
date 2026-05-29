"""FastAPI service for CapScribe.

Ships the extraction output as a queryable internal tool — the "APIs and
internal tools using FastAPI" line item in the JD. Endpoints:

    GET  /health          liveness
    GET  /stats           event counts by type
    GET  /events          list events (optional ?event_type= & ?limit=)
    POST /search          semantic search  {query, k}
    POST /ask             agentic RAG      {question, mode}
    POST /index           rebuild the index from a different extracted JSON

The index is built once on startup from settings.events_path using local
embeddings, so the server boots and serves search/ask with no API spend.
Only POST /ask with mode="llm" calls Claude.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import CapScribeAgent
from config import settings
from retrieval import EventStore, load_events

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("capscribe.api")


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


class SearchRequest(BaseModel):
    query: str
    k: int | None = None


class AskRequest(BaseModel):
    question: str
    mode: str = "extractive"  # "extractive" (free) | "llm" (uses Claude)


class IndexRequest(BaseModel):
    events_path: str


def _build(events_path: str) -> int:
    store = EventStore()  # default local embeddings, persistent
    n = store.index_events(load_events(events_path))
    state["store"] = store
    state["agent"] = CapScribeAgent(store)
    state["source"] = events_path
    logger.info("indexed %d events from %s", n, events_path)
    return n


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "source": state["source"], "indexed": state["store"].count()}


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
    hits = state["store"].search(req.query, k=req.k)
    return {"query": req.query, "hits": [h.model_dump() for h in hits]}


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


@app.get("/")
def home() -> FileResponse:
    return FileResponse("frontend/index.html")
